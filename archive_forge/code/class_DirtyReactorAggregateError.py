from __future__ import annotations
from random import randrange
from typing import Any, Callable, TextIO, TypeVar
from typing_extensions import ParamSpec
from twisted.internet import interfaces, utils
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.lockfile import FilesystemLock
class DirtyReactorAggregateError(Exception):
    """
    Passed to L{twisted.trial.itrial.IReporter.addError} when the reactor is
    left in an unclean state after a test.

    @ivar delayedCalls: The L{DelayedCall<twisted.internet.base.DelayedCall>}
        objects which weren't cleaned up.
    @ivar selectables: The selectables which weren't cleaned up.
    """

    def __init__(self, delayedCalls, selectables=None):
        self.delayedCalls = delayedCalls
        self.selectables = selectables

    def __str__(self) -> str:
        """
        Return a multi-line message describing all of the unclean state.
        """
        msg = 'Reactor was unclean.'
        if self.delayedCalls:
            msg += '\nDelayedCalls: (set twisted.internet.base.DelayedCall.debug = True to debug)\n'
            msg += '\n'.join(map(str, self.delayedCalls))
        if self.selectables:
            msg += '\nSelectables:\n'
            msg += '\n'.join(map(str, self.selectables))
        return msg