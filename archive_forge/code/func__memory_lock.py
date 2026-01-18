import contextlib
import copy
import itertools
import posixpath as pp
import fasteners
from taskflow import exceptions as exc
from taskflow.persistence import path_based
from taskflow.types import tree
@contextlib.contextmanager
def _memory_lock(self, write=False):
    if write:
        lock = self.backend.lock.write_lock
    else:
        lock = self.backend.lock.read_lock
    with lock():
        try:
            yield
        except exc.TaskFlowException:
            raise
        except Exception:
            exc.raise_with_cause(exc.StorageFailure, 'Storage backend internal error')