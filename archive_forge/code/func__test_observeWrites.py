from io import StringIO
from types import TracebackType
from typing import IO, Any, AnyStr, Optional, Type, cast
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from .._file import FileLogObserver, textFileLogObserver
from .._interfaces import ILogObserver
def _test_observeWrites(self, what: Optional[str], count: int) -> None:
    """
        Verify that observer performs an expected number of writes when the
        formatter returns a given value.

        @param what: the value for the formatter to return.
        @param count: the expected number of writes.
        """
    with DummyFile() as fileHandle:
        observer = FileLogObserver(cast(IO[Any], fileHandle), lambda e: what)
        event = dict(x=1)
        observer(event)
        self.assertEqual(fileHandle.writes, count)