import asyncio
import sys
from copy import deepcopy
from collections import defaultdict
import concurrent.futures
from dataclasses import dataclass, field
import logging
import numpy as np
import pprint
import time
import traceback
from typing import Callable, Dict, List, Optional, Tuple, Union
from ray.util.state import list_tasks
import ray
from ray.actor import ActorHandle
from ray.util.state import list_workers
from ray._private.gcs_utils import GcsAioClient, GcsChannel
from ray.util.state.state_manager import StateDataSourceClient
from ray.dashboard.state_aggregator import (
@ray.remote(num_cpus=0)
class StateAPIGeneratorActor:

    def __init__(self, apis: List[StateAPICallSpec], call_interval_s: float=5.0, print_interval_s: float=20.0, wait_after_stop: bool=True, print_result: bool=False) -> None:
        """An actor that periodically issues state API

        Args:
            - apis: List of StateAPICallSpec
            - call_interval_s: State apis in the `apis` will be issued
                every `call_interval_s` seconds.
            - print_interval_s: How frequent state api stats will be dumped.
            - wait_after_stop: When true, call to `ray.get(actor.stop.remote())`
                will wait for all pending state APIs to return.
                Setting it to `False` might miss some long-running state apis calls.
            - print_result: True if result of each API call is printed. Default False.
        """
        self._apis = apis
        self._call_interval_s = call_interval_s
        self._print_interval_s = print_interval_s
        self._wait_after_cancel = wait_after_stop
        self._logger = logging.getLogger(self.__class__.__name__)
        self._print_result = print_result
        self._tasks = None
        self._fut_queue = None
        self._executor = None
        self._loop = None
        self._stopping = False
        self._stopped = False
        self._stats = StateAPIStats()

    async def start(self):
        self._fut_queue = asyncio.Queue()
        self._executor = concurrent.futures.ThreadPoolExecutor()
        self._tasks = [asyncio.ensure_future(awt) for awt in [self._run_generator(), self._run_result_waiter(), self._run_stats_reporter()]]
        await asyncio.gather(*self._tasks)

    def call(self, fn, verify_cb, **kwargs):

        def run_fn():
            try:
                self._logger.debug(f'calling {fn.__name__}({kwargs})')
                return invoke_state_api(verify_cb, fn, state_stats=self._stats, print_result=self._print_result, **kwargs)
            except Exception as e:
                self._logger.warning(f'{fn.__name__}({kwargs}) failed with: {repr(e)}')
                return None
        fut = asyncio.get_running_loop().run_in_executor(self._executor, run_fn)
        return fut

    async def _run_stats_reporter(self):
        while not self._stopped:
            self._logger.info(pprint.pprint(aggregate_perf_results(self._stats)))
            try:
                await asyncio.sleep(self._print_interval_s)
            except asyncio.CancelledError:
                self._logger.info(f'_run_stats_reporter cancelled, waiting for all api {self._stats.pending_calls}calls to return...')

    async def _run_generator(self):
        try:
            while not self._stopping:
                for api_spec in self._apis:
                    fut = self.call(api_spec.api, api_spec.verify_cb, **api_spec.kwargs)
                    self._fut_queue.put_nowait(fut)
                await asyncio.sleep(self._call_interval_s)
        except asyncio.CancelledError:
            self._logger.info('_run_generator cancelled, now stopping...')
            return

    async def _run_result_waiter(self):
        try:
            while not self._stopping:
                fut = await self._fut_queue.get()
                await fut
        except asyncio.CancelledError:
            self._logger.info(f'_run_result_waiter cancelled, cancelling {self._fut_queue.qsize()} pending futures...')
            while not self._fut_queue.empty():
                fut = self._fut_queue.get_nowait()
                if self._wait_after_cancel:
                    await fut
                else:
                    fut.cancel()
            return

    def get_stats(self):
        return aggregate_perf_results(self._stats)

    def ready(self):
        pass

    def stop(self):
        self._stopping = True
        self._logger.debug(f'calling stop, canceling {len(self._tasks)} tasks')
        for task in self._tasks:
            task.cancel()
        self._executor.shutdown(wait=self._wait_after_cancel)
        self._stopped = True