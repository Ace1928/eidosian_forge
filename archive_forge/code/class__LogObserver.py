import inspect
import os
import sys
import tempfile
import types
import unittest as pyunit
import warnings
from dis import findlinestarts as _findlinestarts
from typing import (
from unittest import SkipTest
from attrs import frozen
from typing_extensions import ParamSpec
from twisted.internet.defer import Deferred, ensureDeferred
from twisted.python import failure, log, monkey
from twisted.python.deprecate import (
from twisted.python.reflect import fullyQualifiedName
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import itrial, util
class _LogObserver:
    """
    Observes the Twisted logs and catches any errors.

    @ivar _errors: A C{list} of L{Failure} instances which were received as
        error events from the Twisted logging system.

    @ivar _added: A C{int} giving the number of times C{_add} has been called
        less the number of times C{_remove} has been called; used to only add
        this observer to the Twisted logging since once, regardless of the
        number of calls to the add method.

    @ivar _ignored: A C{list} of exception types which will not be recorded.
    """

    def __init__(self):
        self._errors = []
        self._added = 0
        self._ignored = []

    def _add(self):
        if self._added == 0:
            log.addObserver(self.gotEvent)
        self._added += 1

    def _remove(self):
        self._added -= 1
        if self._added == 0:
            log.removeObserver(self.gotEvent)

    def _ignoreErrors(self, *errorTypes):
        """
        Do not store any errors with any of the given types.
        """
        self._ignored.extend(errorTypes)

    def _clearIgnores(self):
        """
        Stop ignoring any errors we might currently be ignoring.
        """
        self._ignored = []

    def flushErrors(self, *errorTypes):
        """
        Flush errors from the list of caught errors. If no arguments are
        specified, remove all errors. If arguments are specified, only remove
        errors of those types from the stored list.
        """
        if errorTypes:
            flushed = []
            remainder = []
            for f in self._errors:
                if f.check(*errorTypes):
                    flushed.append(f)
                else:
                    remainder.append(f)
            self._errors = remainder
        else:
            flushed = self._errors
            self._errors = []
        return flushed

    def getErrors(self):
        """
        Return a list of errors caught by this observer.
        """
        return self._errors

    def gotEvent(self, event):
        """
        The actual observer method. Called whenever a message is logged.

        @param event: A dictionary containing the log message. Actual
        structure undocumented (see source for L{twisted.python.log}).
        """
        if event.get('isError', False) and 'failure' in event:
            f = event['failure']
            if len(self._ignored) == 0 or not f.check(*self._ignored):
                self._errors.append(f)