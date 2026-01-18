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
class GSVSequenceTests(NMEAReceiverSetup, TestCase):
    """
    Tests for the interpretation of GSV sequences.
    """

    def test_firstSentence(self) -> None:
        """
        The first sentence in a GSV sequence is correctly identified.
        """
        self.protocol.lineReceived(GPGSV_FIRST)
        sentence = self.receiver.receivedSentence
        assert sentence is not None
        self.assertTrue(sentence._isFirstGSVSentence())
        self.assertFalse(sentence._isLastGSVSentence())

    def test_middleSentence(self) -> None:
        """
        A sentence in the middle of a GSV sequence is correctly
        identified (as being neither the last nor the first).
        """
        self.protocol.lineReceived(GPGSV_MIDDLE)
        sentence = self.receiver.receivedSentence
        assert sentence is not None
        self.assertFalse(sentence._isFirstGSVSentence())
        self.assertFalse(sentence._isLastGSVSentence())

    def test_lastSentence(self) -> None:
        """
        The last sentence in a GSV sequence is correctly identified.
        """
        self.protocol.lineReceived(GPGSV_LAST)
        sentence = self.receiver.receivedSentence
        assert sentence is not None
        self.assertFalse(sentence._isFirstGSVSentence())
        self.assertTrue(sentence._isLastGSVSentence())