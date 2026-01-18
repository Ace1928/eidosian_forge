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
class BogusSentenceTests(NMEAReceiverSetup, TestCase):
    """
    Tests for verifying predictable failure for bogus NMEA sentences.
    """

    def assertRaisesOnSentence(self, exceptionClass: type[Exception], sentence: str | bytes) -> None:
        """
        Asserts that the protocol raises C{exceptionClass} when it receives
        C{sentence}.

        @param exceptionClass: The exception class expected to be raised.
        @type exceptionClass: C{Exception} subclass

        @param sentence: The (bogus) NMEA sentence.
        @type sentence: C{str}
        """
        self.assertRaises(exceptionClass, self.protocol.lineReceived, sentence)

    def test_raiseOnUnknownSentenceType(self) -> None:
        """
        Receiving a well-formed sentence of unknown type raises
        C{ValueError}.
        """
        self.assertRaisesOnSentence(ValueError, b'$GPBOGUS*5b')

    def test_raiseOnMalformedSentences(self) -> None:
        """
        Receiving a malformed sentence raises L{base.InvalidSentence}.
        """
        self.assertRaisesOnSentence(base.InvalidSentence, 'GPBOGUS')