from __future__ import annotations
import datetime
from operator import attrgetter
from typing import Callable, Iterable, TypedDict
from zope.interface import implementer
from constantly import NamedConstant
from typing_extensions import Literal, Protocol
from twisted.positioning import base, ipositioning, nmea
from twisted.positioning.base import Angles
from twisted.positioning.test.receiver import MockPositioningReceiver
from twisted.trial.unittest import TestCase
class FixerTestMixin:
    """
    Mixin for tests for the fixers on L{nmea.NMEAAdapter} that adapt
    from NMEA-specific notations to generic Python objects.

    @ivar adapter: The NMEA adapter.
    @type adapter: L{nmea.NMEAAdapter}
    """

    def setUp(self) -> None:
        self.adapter = nmea.NMEAAdapter(base.BasePositioningReceiver())

    def _fixerTest(self: _FixerTestMixinBase, sentenceData: dict[str, str], expected: _State | None=None, exceptionClass: type[Exception] | None=None) -> None:
        """
        A generic adapter fixer test.

        Creates a sentence from the C{sentenceData} and sends that to the
        adapter. If C{exceptionClass} is not passed, this is assumed to work,
        and C{expected} is compared with the adapter's internal state.
        Otherwise, passing the sentence to the adapter is checked to raise
        C{exceptionClass}.

        @param sentenceData: Raw sentence content.
        @type sentenceData: C{dict} mapping C{str} to C{str}

        @param expected: The expected state of the adapter.
        @type expected: C{dict} or L{None}

        @param exceptionClass: The exception to be raised by the adapter.
        @type exceptionClass: subclass of C{Exception}
        """
        sentence = nmea.NMEASentence(sentenceData)

        def receiveSentence() -> None:
            self.adapter.sentenceReceived(sentence)
        if exceptionClass is None:
            receiveSentence()
            self.assertEqual(self.adapter._state, expected)
        else:
            self.assertRaises(exceptionClass, receiveSentence)
        self.adapter.clear()