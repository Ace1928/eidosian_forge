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