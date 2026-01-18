from __future__ import annotations
import sys
import warnings
from io import StringIO
from typing import Mapping, Sequence, TypeVar
from unittest import TestResult
from twisted.python.filepath import FilePath
from twisted.trial._synctest import (
from twisted.trial.unittest import SynchronousTestCase
import warnings
import warnings
class CollectWarningsTests(SynchronousTestCase):
    """
    Tests for L{_collectWarnings}.
    """

    def test_callsObserver(self) -> None:
        """
        L{_collectWarnings} calls the observer with each emitted warning.
        """
        firstMessage = 'dummy calls observer warning'
        secondMessage = firstMessage[::-1]
        thirdMessage = Warning(1, 2, 3)
        events: list[str | _Warning] = []

        def f() -> None:
            events.append('call')
            warnings.warn(firstMessage)
            warnings.warn(secondMessage)
            warnings.warn(thirdMessage)
            events.append('returning')
        _collectWarnings(events.append, f)
        self.assertEqual(events[0], 'call')
        assert isinstance(events[1], _Warning)
        self.assertEqual(events[1].message, firstMessage)
        assert isinstance(events[2], _Warning)
        self.assertEqual(events[2].message, secondMessage)
        assert isinstance(events[3], _Warning)
        self.assertEqual(events[3].message, str(thirdMessage))
        self.assertEqual(events[4], 'returning')
        self.assertEqual(len(events), 5)

    def test_suppresses(self) -> None:
        """
        Any warnings emitted by a call to a function passed to
        L{_collectWarnings} are not actually emitted to the warning system.
        """
        output = StringIO()
        self.patch(sys, 'stdout', output)
        _collectWarnings(lambda x: None, warnings.warn, 'text')
        self.assertEqual(output.getvalue(), '')

    def test_callsFunction(self) -> None:
        """
        L{_collectWarnings} returns the result of calling the callable passed to
        it with the parameters given.
        """
        arguments = []
        value = object()

        def f(*args: object, **kwargs: object) -> object:
            arguments.append((args, kwargs))
            return value
        result = _collectWarnings(lambda x: None, f, 1, 'a', b=2, c='d')
        self.assertEqual(arguments, [((1, 'a'), {'b': 2, 'c': 'd'})])
        self.assertIdentical(result, value)

    def test_duplicateWarningCollected(self) -> None:
        """
        Subsequent emissions of a warning from a particular source site can be
        collected by L{_collectWarnings}.  In particular, the per-module
        emitted-warning cache should be bypassed (I{__warningregistry__}).
        """
        global __warningregistry__
        del __warningregistry__

        def f() -> None:
            warnings.warn('foo')
        warnings.simplefilter('default')
        f()
        events: list[_Warning] = []
        _collectWarnings(events.append, f)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].message, 'foo')
        self.assertEqual(len(self.flushWarnings()), 1)

    def test_immutableObject(self) -> None:
        """
        L{_collectWarnings}'s behavior is not altered by the presence of an
        object which cannot have attributes set on it as a value in
        C{sys.modules}.
        """
        key = object()
        sys.modules[key] = key
        self.addCleanup(sys.modules.pop, key)
        self.test_duplicateWarningCollected()

    def test_setWarningRegistryChangeWhileIterating(self) -> None:
        """
        If the dictionary passed to L{_setWarningRegistryToNone} changes size
        partway through the process, C{_setWarningRegistryToNone} continues to
        set C{__warningregistry__} to L{None} on the rest of the values anyway.


        This might be caused by C{sys.modules} containing something that's not
        really a module and imports things on setattr.  py.test does this, as
        does L{twisted.python.deprecate.deprecatedModuleAttribute}.
        """
        d: dict[object, A | None] = {}

        class A:

            def __init__(self, key: object) -> None:
                self.__dict__['_key'] = key

            def __setattr__(self, value: object, item: object) -> None:
                d[self._key] = None
        key1 = object()
        key2 = object()
        d[key1] = A(key2)
        key3 = object()
        key4 = object()
        d[key3] = A(key4)
        _setWarningRegistryToNone(d)
        self.assertEqual({key1, key2, key3, key4}, set(d.keys()))