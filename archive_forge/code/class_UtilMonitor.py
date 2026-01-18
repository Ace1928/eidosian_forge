import copy
import glob
import inspect
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime
from numbers import Number
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
import numpy as np
import psutil
import ray
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.air._internal.json import SafeFallbackEncoder  # noqa
from ray.air._internal.util import (  # noqa: F401
from ray._private.dict import (  # noqa: F401
@DeveloperAPI
class UtilMonitor(Thread):
    """Class for system usage utilization monitoring.

    It keeps track of CPU, RAM, GPU, VRAM usage (each gpu separately) by
    pinging for information every x seconds in a separate thread.

    Requires psutil and GPUtil to be installed. Can be enabled with
    Tuner(param_space={"log_sys_usage": True}).
    """

    def __init__(self, start=True, delay=0.7):
        self.stopped = True
        GPUtil = _import_gputil()
        self.GPUtil = GPUtil
        if GPUtil is None and start:
            logger.warning('Install gputil for GPU system monitoring.')
        if psutil is None and start:
            logger.warning('Install psutil to monitor system performance.')
        if GPUtil is None and psutil is None:
            return
        super(UtilMonitor, self).__init__()
        self.delay = delay
        self.values = defaultdict(list)
        self.lock = threading.Lock()
        self.daemon = True
        if start:
            self.start()

    def _read_utilization(self):
        with self.lock:
            if psutil is not None:
                self.values['cpu_util_percent'].append(float(psutil.cpu_percent(interval=None)))
                self.values['ram_util_percent'].append(float(getattr(psutil.virtual_memory(), 'percent')))
            if self.GPUtil is not None:
                gpu_list = []
                try:
                    gpu_list = self.GPUtil.getGPUs()
                except Exception:
                    logger.debug('GPUtil failed to retrieve GPUs.')
                for gpu in gpu_list:
                    self.values['gpu_util_percent' + str(gpu.id)].append(float(gpu.load))
                    self.values['vram_util_percent' + str(gpu.id)].append(float(gpu.memoryUtil))

    def get_data(self):
        if self.stopped:
            return {}
        with self.lock:
            ret_values = copy.deepcopy(self.values)
            for key, val in self.values.items():
                del val[:]
        return {'perf': {k: np.mean(v) for k, v in ret_values.items() if len(v) > 0}}

    def run(self):
        self.stopped = False
        while not self.stopped:
            self._read_utilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True