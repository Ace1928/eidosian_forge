import collections
import contextlib
import itertools
import threading
from automaton import runners
from concurrent import futures
import fasteners
import functools
import networkx as nx
from oslo_utils import excutils
from oslo_utils import strutils
from oslo_utils import timeutils
from taskflow.engines.action_engine import builder
from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import executor
from taskflow.engines.action_engine import process_executor
from taskflow.engines.action_engine import runtime
from taskflow.engines import base
from taskflow import exceptions as exc
from taskflow import logging
from taskflow import states
from taskflow import storage
from taskflow.types import failure
from taskflow.utils import misc
class ParallelActionEngine(ActionEngine):
    """Engine that runs tasks in parallel manner.

    **Additional engine options:**

    * ``executor``: a object that implements a :pep:`3148` compatible executor
      interface; it will be used for scheduling tasks. The following
      type are applicable (other unknown types passed will cause a type
      error to be raised).

=========================  ===============================================
Type provided              Executor used
=========================  ===============================================
|cft|.ThreadPoolExecutor   :class:`~.executor.ParallelThreadTaskExecutor`
|cfp|.ProcessPoolExecutor  :class:`~.|pe|.ParallelProcessTaskExecutor`
|cf|._base.Executor        :class:`~.executor.ParallelThreadTaskExecutor`
=========================  ===============================================

    * ``executor``: a string that will be used to select a :pep:`3148`
      compatible executor; it will be used for scheduling tasks. The following
      string are applicable (other unknown strings passed will cause a value
      error to be raised).

===========================  ===============================================
String (case insensitive)    Executor used
===========================  ===============================================
``process``                  :class:`~.|pe|.ParallelProcessTaskExecutor`
``processes``                :class:`~.|pe|.ParallelProcessTaskExecutor`
``thread``                   :class:`~.executor.ParallelThreadTaskExecutor`
``threaded``                 :class:`~.executor.ParallelThreadTaskExecutor`
``threads``                  :class:`~.executor.ParallelThreadTaskExecutor`
``greenthread``              :class:`~.executor.ParallelThreadTaskExecutor`
                              (greened version)
``greedthreaded``            :class:`~.executor.ParallelThreadTaskExecutor`
                              (greened version)
``greenthreads``             :class:`~.executor.ParallelThreadTaskExecutor`
                              (greened version)
===========================  ===============================================

    * ``max_workers``: a integer that will affect the number of parallel
      workers that are used to dispatch tasks into (this number is bounded
      by the maximum parallelization your workflow can support).

    * ``wait_timeout``: a float (in seconds) that will affect the
      parallel process task executor (and therefore is **only** applicable when
      the executor provided above is of the process variant). This number
      affects how much time the process task executor waits for messages from
      child processes (typically indicating they have finished or failed). A
      lower number will have high granularity but *currently* involves more
      polling while a higher number will involve less polling but a slower time
      for an engine to notice a task has completed.

    .. |pe|  replace:: process_executor
    .. |cfp| replace:: concurrent.futures.process
    .. |cft| replace:: concurrent.futures.thread
    .. |cf| replace:: concurrent.futures
    """
    _executor_cls_matchers = [_ExecutorTypeMatch((futures.ThreadPoolExecutor,), executor.ParallelThreadTaskExecutor), _ExecutorTypeMatch((futures.ProcessPoolExecutor,), process_executor.ParallelProcessTaskExecutor), _ExecutorTypeMatch((futures.Executor,), executor.ParallelThreadTaskExecutor)]
    _executor_str_matchers = [_ExecutorTextMatch(frozenset(['processes', 'process']), process_executor.ParallelProcessTaskExecutor), _ExecutorTextMatch(frozenset(['thread', 'threads', 'threaded']), executor.ParallelThreadTaskExecutor), _ExecutorTextMatch(frozenset(['greenthread', 'greenthreads', 'greenthreaded']), executor.ParallelGreenThreadTaskExecutor)]
    _default_executor_cls = executor.ParallelThreadTaskExecutor

    def __init__(self, flow, flow_detail, backend, options):
        super(ParallelActionEngine, self).__init__(flow, flow_detail, backend, options)
        self._task_executor = self._fetch_task_executor(self._options)

    @classmethod
    def _fetch_task_executor(cls, options):
        kwargs = {}
        executor_cls = cls._default_executor_cls
        desired_executor = options.get('executor')
        if isinstance(desired_executor, str):
            matched_executor_cls = None
            for m in cls._executor_str_matchers:
                if m.matches(desired_executor):
                    matched_executor_cls = m.executor_cls
                    break
            if matched_executor_cls is None:
                expected = set()
                for m in cls._executor_str_matchers:
                    expected.update(m.strings)
                raise ValueError("Unknown executor string '%s' expected one of %s (or mixed case equivalent)" % (desired_executor, list(expected)))
            else:
                executor_cls = matched_executor_cls
        elif desired_executor is not None:
            matched_executor_cls = None
            for m in cls._executor_cls_matchers:
                if m.matches(desired_executor):
                    matched_executor_cls = m.executor_cls
                    break
            if matched_executor_cls is None:
                expected = set()
                for m in cls._executor_cls_matchers:
                    expected.update(m.types)
                raise TypeError("Unknown executor '%s' (%s) expected an instance of %s" % (desired_executor, type(desired_executor), list(expected)))
            else:
                executor_cls = matched_executor_cls
                kwargs['executor'] = desired_executor
        try:
            for k, value_converter in executor_cls.constructor_options:
                try:
                    kwargs[k] = value_converter(options[k])
                except KeyError:
                    pass
        except AttributeError:
            pass
        return executor_cls(**kwargs)