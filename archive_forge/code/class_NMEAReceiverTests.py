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
class NMEAReceiverTests(TestCase):
    """
    Tests for the NMEA receiver.
    """

    def setUp(self) -> None:
        self.receiver = MockPositioningReceiver()
        self.adapter = nmea.NMEAAdapter(self.receiver)
        self.protocol = nmea.NMEAProtocol(self.adapter)

    def test_onlyFireWhenCurrentSentenceHasNewInformation(self) -> None:
        """
        If the current sentence does not contain any new fields for a
        particular callback, that callback is not called; even if all
        necessary information is still in the state from one or more
        previous messages.
        """
        self.protocol.lineReceived(GPGGA)
        gpggaCallbacks = {'positionReceived', 'positionErrorReceived', 'altitudeReceived'}
        self.assertEqual(set(self.receiver.called.keys()), gpggaCallbacks)
        self.receiver.clear()
        self.assertNotEqual(self.adapter._state, {})
        self.protocol.lineReceived(GPHDT)
        gphdtCallbacks = {'headingReceived'}
        self.assertEqual(set(self.receiver.called.keys()), gphdtCallbacks)

    def _receiverTest(self, sentences: Iterable[bytes], expectedFired: Iterable[str]=(), extraTest: Callable[[], None] | None=None) -> None:
        """
        A generic test for NMEA receiver behavior.

        @param sentences: The sequence of sentences to simulate receiving.
        @type sentences: iterable of C{str}
        @param expectedFired: The names of the callbacks expected to fire.
        @type expectedFired: iterable of C{str}
        @param extraTest: An optional extra test hook.
        @type extraTest: nullary callable
        """
        for sentence in sentences:
            self.protocol.lineReceived(sentence)
        actuallyFired = self.receiver.called.keys()
        self.assertEqual(set(actuallyFired), set(expectedFired))
        if extraTest is not None:
            extraTest()
        self.receiver.clear()
        self.adapter.clear()

    def test_positionErrorUpdateAcrossStates(self) -> None:
        """
        The positioning error is updated across multiple states.
        """
        sentences = [GPGSA] + GPGSV_SEQ
        callbacksFired = ['positionErrorReceived', 'beaconInformationReceived']

        def _getIdentifiers(beacons: Iterable[base.Satellite]) -> list[int]:
            return sorted(map(attrgetter('identifier'), beacons))

        def checkBeaconInformation() -> None:
            beaconInformation = self.adapter._state['beaconInformation']
            seenIdentifiers = _getIdentifiers(beaconInformation.seenBeacons)
            expected = [3, 4, 6, 13, 14, 16, 18, 19, 22, 24, 27]
            self.assertEqual(seenIdentifiers, expected)
            usedIdentifiers = _getIdentifiers(beaconInformation.usedBeacons)
            self.assertEqual(usedIdentifiers, [14, 18, 19, 22, 27])
        self._receiverTest(sentences, callbacksFired, checkBeaconInformation)

    def test_emptyMiddleGSV(self) -> None:
        """
        A GSV sentence with empty entries in any position does not mean that
        entries in subsequent positions of the same GSV sentence are ignored.
        """
        sentences = [GPGSV_EMPTY_MIDDLE]
        callbacksFired = ['beaconInformationReceived']

        def checkBeaconInformation() -> None:
            beaconInformation = self.adapter._state['beaconInformation']
            seenBeacons = beaconInformation.seenBeacons
            self.assertEqual(len(seenBeacons), 2)
            self.assertIn(13, [b.identifier for b in seenBeacons])
        self._receiverTest(sentences, callbacksFired, checkBeaconInformation)

    def test_GGASentences(self) -> None:
        """
        A sequence of GGA sentences fires C{positionReceived},
        C{positionErrorReceived} and C{altitudeReceived}.
        """
        sentences = [GPGGA]
        callbacksFired = ['positionReceived', 'positionErrorReceived', 'altitudeReceived']
        self._receiverTest(sentences, callbacksFired)

    def test_GGAWithDateInState(self) -> None:
        """
        When receiving a GPGGA sentence and a date was already in the
        state, the new time (from the GPGGA sentence) is combined with
        that date.
        """
        self.adapter._state['_date'] = datetime.date(2014, 1, 1)
        sentences = [GPGGA]
        callbacksFired = ['positionReceived', 'positionErrorReceived', 'altitudeReceived', 'timeReceived']
        self._receiverTest(sentences, callbacksFired)

    def test_RMCSentences(self) -> None:
        """
        A sequence of RMC sentences fires C{positionReceived},
        C{speedReceived}, C{headingReceived} and C{timeReceived}.
        """
        sentences = [GPRMC]
        callbacksFired = ['headingReceived', 'speedReceived', 'positionReceived', 'timeReceived']
        self._receiverTest(sentences, callbacksFired)

    def test_GSVSentences(self) -> None:
        """
        A complete sequence of GSV sentences fires
        C{beaconInformationReceived}.
        """
        sentences = [GPGSV_FIRST, GPGSV_MIDDLE, GPGSV_LAST]
        callbacksFired = ['beaconInformationReceived']

        def checkPartialInformation() -> None:
            self.assertNotIn('_partialBeaconInformation', self.adapter._state)
        self._receiverTest(sentences, callbacksFired, checkPartialInformation)

    def test_emptyMiddleEntriesGSVSequence(self) -> None:
        """
        A complete sequence of GSV sentences with empty entries in the
        middle still fires C{beaconInformationReceived}.
        """
        sentences = [GPGSV_EMPTY_MIDDLE]
        self._receiverTest(sentences, ['beaconInformationReceived'])

    def test_incompleteGSVSequence(self) -> None:
        """
        An incomplete sequence of GSV sentences does not fire any callbacks.
        """
        sentences = [GPGSV_FIRST]
        self._receiverTest(sentences)

    def test_singleSentenceGSVSequence(self) -> None:
        """
        The parser does not fail badly when the sequence consists of
        only one sentence (but is otherwise complete).
        """
        sentences = [GPGSV_SINGLE]
        self._receiverTest(sentences, ['beaconInformationReceived'])

    def test_GLLSentences(self) -> None:
        """
        GLL sentences fire C{positionReceived}.
        """
        sentences = [GPGLL_PARTIAL, GPGLL]
        self._receiverTest(sentences, ['positionReceived'])

    def test_HDTSentences(self) -> None:
        """
        HDT sentences fire C{headingReceived}.
        """
        sentences = [GPHDT]
        self._receiverTest(sentences, ['headingReceived'])

    def test_mixedSentences(self) -> None:
        """
        A mix of sentences fires the correct callbacks.
        """
        sentences = [GPRMC, GPGGA]
        callbacksFired = ['altitudeReceived', 'speedReceived', 'positionReceived', 'positionErrorReceived', 'timeReceived', 'headingReceived']

        def checkTime() -> None:
            expectedDateTime = datetime.datetime(1994, 3, 23, 12, 35, 19)
            self.assertEqual(self.adapter._state['time'], expectedDateTime)
        self._receiverTest(sentences, callbacksFired, checkTime)

    def test_lotsOfMixedSentences(self) -> None:
        """
        Sends an entire gamut of sentences and verifies the
        appropriate callbacks fire. These are more than you'd expect
        from your average consumer GPS device. They have most of the
        important information, including beacon information and
        visibility.
        """
        sentences = [GPGSA] + GPGSV_SEQ + [GPRMC, GPGGA, GPGLL]
        callbacksFired = ['headingReceived', 'beaconInformationReceived', 'speedReceived', 'positionReceived', 'timeReceived', 'altitudeReceived', 'positionErrorReceived']
        self._receiverTest(sentences, callbacksFired)