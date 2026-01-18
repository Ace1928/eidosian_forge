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
def _pre_check(check_compiled=True, check_storage_ensured=True, check_validated=True):
    """Engine state precondition checking decorator."""

    def decorator(meth):
        do_what = meth.__name__

        @functools.wraps(meth)
        def wrapper(self, *args, **kwargs):
            if check_compiled and (not self._compiled):
                raise exc.InvalidState('Can not %s an engine which has not been compiled' % do_what)
            if check_storage_ensured and (not self._storage_ensured):
                raise exc.InvalidState('Can not %s an engine which has not had its storage populated' % do_what)
            if check_validated and (not self._validated):
                raise exc.InvalidState('Can not %s an engine which has not been validated' % do_what)
            return meth(self, *args, **kwargs)
        return wrapper
    return decorator