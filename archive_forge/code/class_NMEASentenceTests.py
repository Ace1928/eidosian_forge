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
class NMEASentenceTests(NMEAReceiverSetup, TestCase):
    """
    Tests for L{nmea.NMEASentence} objects.
    """

    def test_repr(self) -> None:
        """
        The C{repr} of L{nmea.NMEASentence} objects is correct.
        """
        sentencesWithExpectedRepr = [(GPGSA, '<NMEASentence (GPGSA) {dataMode: A, fixType: 3, horizontalDilutionOfPrecision: 1.0, positionDilutionOfPrecision: 1.7, usedSatellitePRN_0: 19, usedSatellitePRN_1: 28, usedSatellitePRN_2: 14, usedSatellitePRN_3: 18, usedSatellitePRN_4: 27, usedSatellitePRN_5: 22, usedSatellitePRN_6: 31, usedSatellitePRN_7: 39, verticalDilutionOfPrecision: 1.3}>')]
        for sentence, expectedRepr in sentencesWithExpectedRepr:
            self.protocol.lineReceived(sentence)
            received = self.receiver.receivedSentence
            self.assertEqual(repr(received), expectedRepr)