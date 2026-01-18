import errno
from io import StringIO
from signal import SIGTERM
from types import TracebackType
from typing import Any, Iterable, List, Optional, TextIO, Tuple, Type, Union, cast
from attr import Factory, attrib, attrs
import twisted.trial.unittest
from twisted.internet.testing import MemoryReactor
from twisted.logger import (
from twisted.python.filepath import FilePath
from ...runner import _runner
from .._exit import ExitStatus
from .._pidfile import NonePIDFile, PIDFile
from .._runner import Runner
def _testHook(self, methodName: str, callerName: Optional[str]=None) -> None:
    """
        Verify that the named hook is run with the expected arguments as
        specified by the arguments used to create the L{Runner}, when the
        specified caller is invoked.

        @param methodName: The name of the hook to verify.

        @param callerName: The name of the method that is expected to cause the
            hook to be called.
            If C{None}, use the L{Runner} method with the same name as the
            hook.
        """
    if callerName is None:
        callerName = methodName
    arguments = dict(a=object(), b=object(), c=object())
    argumentsSeen = []

    def hook(**arguments: object) -> None:
        argumentsSeen.append(arguments)
    runnerArguments = {methodName: hook, f'{methodName}Arguments': arguments.copy()}
    runner = Runner(reactor=MemoryReactor(), **runnerArguments)
    hookCaller = getattr(runner, callerName)
    hookCaller()
    self.assertEqual(len(argumentsSeen), 1)
    self.assertEqual(argumentsSeen[0], arguments)