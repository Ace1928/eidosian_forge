import os
import sys
import time
import traceback
from math import sqrt
from typing import Any, Callable, Dict, List, Optional, Sequence
from sphinx.errors import SphinxParallelError
from sphinx.util import logging
class ParallelTasks:
    """Executes *nproc* tasks in parallel after forking."""

    def __init__(self, nproc: int) -> None:
        self.nproc = nproc
        self._result_funcs: Dict[int, Callable] = {}
        self._args: Dict[int, Optional[List[Any]]] = {}
        self._procs: Dict[int, ForkProcess] = {}
        self._precvs: Dict[int, Any] = {}
        self._precvsWaiting: Dict[int, Any] = {}
        self._pworking = 0
        self._taskid = 0

    def _process(self, pipe: Any, func: Callable, arg: Any) -> None:
        try:
            collector = logging.LogCollector()
            with collector.collect():
                if arg is None:
                    ret = func()
                else:
                    ret = func(arg)
            failed = False
        except BaseException as err:
            failed = True
            errmsg = traceback.format_exception_only(err.__class__, err)[0].strip()
            ret = (errmsg, traceback.format_exc())
        logging.convert_serializable(collector.logs)
        pipe.send((failed, collector.logs, ret))

    def add_task(self, task_func: Callable, arg: Any=None, result_func: Optional[Callable]=None) -> None:
        tid = self._taskid
        self._taskid += 1
        self._result_funcs[tid] = result_func or (lambda arg, result: None)
        self._args[tid] = arg
        precv, psend = multiprocessing.Pipe(False)
        context: ForkContext = multiprocessing.get_context('fork')
        proc = context.Process(target=self._process, args=(psend, task_func, arg))
        self._procs[tid] = proc
        self._precvsWaiting[tid] = precv
        self._join_one()

    def join(self) -> None:
        try:
            while self._pworking:
                if not self._join_one():
                    time.sleep(0.02)
        except Exception:
            self.terminate()
            raise

    def terminate(self) -> None:
        for tid in list(self._precvs):
            self._procs[tid].terminate()
            self._result_funcs.pop(tid)
            self._procs.pop(tid)
            self._precvs.pop(tid)
            self._pworking -= 1

    def _join_one(self) -> bool:
        joined_any = False
        for tid, pipe in self._precvs.items():
            if pipe.poll():
                exc, logs, result = pipe.recv()
                if exc:
                    raise SphinxParallelError(*result)
                for log in logs:
                    logger.handle(log)
                self._result_funcs.pop(tid)(self._args.pop(tid), result)
                self._procs[tid].join()
                self._precvs.pop(tid)
                self._pworking -= 1
                joined_any = True
                break
        while self._precvsWaiting and self._pworking < self.nproc:
            newtid, newprecv = self._precvsWaiting.popitem()
            self._precvs[newtid] = newprecv
            self._procs[newtid].start()
            self._pworking += 1
        return joined_any