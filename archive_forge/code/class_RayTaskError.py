import os
from traceback import format_exception
from typing import Optional, Union
import colorama
import ray._private.ray_constants as ray_constants
import ray.cloudpickle as pickle
from ray._raylet import ActorID, TaskID, WorkerID
from ray.core.generated.common_pb2 import (
from ray.util.annotations import DeveloperAPI, PublicAPI
import setproctitle
@PublicAPI
class RayTaskError(RayError):
    """Indicates that a task threw an exception during execution.

    If a task throws an exception during execution, a RayTaskError is stored in
    the object store for each of the task's outputs. When an object is
    retrieved from the object store, the Python method that retrieved it checks
    to see if the object is a RayTaskError and if it is then an exception is
    thrown propagating the error message.
    """

    def __init__(self, function_name, traceback_str, cause, proctitle=None, pid=None, ip=None, actor_repr=None, actor_id=None):
        """Initialize a RayTaskError."""
        import ray
        self.args = (function_name, traceback_str, cause, proctitle, pid, ip)
        if proctitle:
            self.proctitle = proctitle
        else:
            self.proctitle = setproctitle.getproctitle()
        self.pid = pid or os.getpid()
        self.ip = ip or ray.util.get_node_ip_address()
        self.function_name = function_name
        self.traceback_str = traceback_str
        self.actor_repr = actor_repr
        self._actor_id = actor_id
        self.cause = cause
        assert traceback_str is not None

    def as_instanceof_cause(self):
        """Returns an exception that is an instance of the cause's class.

        The returned exception will inherit from both RayTaskError and the
        cause class and will contain all of the attributes of the cause
        exception.
        """
        cause_cls = self.cause.__class__
        if issubclass(RayTaskError, cause_cls):
            return self
        error_msg = str(self)

        class cls(RayTaskError, cause_cls):

            def __init__(self, cause):
                self.cause = cause
                self.args = (cause,)

            def __getattr__(self, name):
                return getattr(self.cause, name)

            def __str__(self):
                return error_msg
        name = f'RayTaskError({cause_cls.__name__})'
        cls.__name__ = name
        cls.__qualname__ = name
        return cls(self.cause)

    def __str__(self):
        """Format a RayTaskError as a string."""
        lines = self.traceback_str.strip().split('\n')
        out = []
        code_from_internal_file = False
        for i, line in enumerate(lines):
            if line.startswith('Traceback '):
                traceback_line = f'{colorama.Fore.CYAN}{self.proctitle}(){colorama.Fore.RESET} (pid={self.pid}, ip={self.ip}'
                if self.actor_repr:
                    traceback_line += f', actor_id={self._actor_id}, repr={self.actor_repr})'
                else:
                    traceback_line += ')'
                code_from_internal_file = False
                out.append(traceback_line)
            elif line.startswith('  File ') and ('ray/worker.py' in line or 'ray/_private/' in line or 'ray/util/tracing/' in line or ('ray/_raylet.pyx' in line)):
                if 'ray._raylet.raise_if_dependency_failed' in line:
                    out.append('  At least one of the input arguments for this task could not be computed:')
                if i + 1 < len(lines) and lines[i + 1].startswith('    '):
                    code_from_internal_file = True
            elif code_from_internal_file:
                code_from_internal_file = False
            else:
                out.append(line)
        return '\n'.join(out)