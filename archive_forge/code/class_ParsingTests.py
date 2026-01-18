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
class ParsingTests(NMEAReceiverSetup, TestCase):
    """
    Tests if raw NMEA sentences get parsed correctly.

    This doesn't really involve any interpretation, just turning ugly raw NMEA
    representations into objects that are more pleasant to work with.
    """

    def _parserTest(self, sentence: bytes, expected: dict[str, str]) -> None:
        """
        Passes a sentence to the protocol and gets the parsed sentence from
        the receiver. Then verifies that the parsed sentence contains the
        expected data.
        """
        self.protocol.lineReceived(sentence)
        received = self.receiver.receivedSentence
        assert received is not None
        self.assertEqual(expected, received._sentenceData)

    def test_fullRMC(self) -> None:
        """
        A full RMC sentence is correctly parsed.
        """
        expected = {'type': 'GPRMC', 'latitudeFloat': '4807.038', 'latitudeHemisphere': 'N', 'longitudeFloat': '01131.000', 'longitudeHemisphere': 'E', 'magneticVariation': '003.1', 'magneticVariationDirection': 'W', 'speedInKnots': '022.4', 'timestamp': '123519', 'datestamp': '230394', 'trueHeading': '084.4', 'dataMode': 'A'}
        self._parserTest(GPRMC, expected)

    def test_fullGGA(self) -> None:
        """
        A full GGA sentence is correctly parsed.
        """
        expected = {'type': 'GPGGA', 'altitude': '545.4', 'altitudeUnits': 'M', 'heightOfGeoidAboveWGS84': '46.9', 'heightOfGeoidAboveWGS84Units': 'M', 'horizontalDilutionOfPrecision': '0.9', 'latitudeFloat': '4807.038', 'latitudeHemisphere': 'N', 'longitudeFloat': '01131.000', 'longitudeHemisphere': 'E', 'numberOfSatellitesSeen': '08', 'timestamp': '123519', 'fixQuality': '1'}
        self._parserTest(GPGGA, expected)

    def test_fullGLL(self) -> None:
        """
        A full GLL sentence is correctly parsed.
        """
        expected = {'type': 'GPGLL', 'latitudeFloat': '4916.45', 'latitudeHemisphere': 'N', 'longitudeFloat': '12311.12', 'longitudeHemisphere': 'W', 'timestamp': '225444', 'dataMode': 'A'}
        self._parserTest(GPGLL, expected)

    def test_partialGLL(self) -> None:
        """
        A partial GLL sentence is correctly parsed.
        """
        expected = {'type': 'GPGLL', 'latitudeFloat': '3751.65', 'latitudeHemisphere': 'S', 'longitudeFloat': '14507.36', 'longitudeHemisphere': 'E'}
        self._parserTest(GPGLL_PARTIAL, expected)

    def test_fullGSV(self) -> None:
        """
        A full GSV sentence is correctly parsed.
        """
        expected = {'type': 'GPGSV', 'GSVSentenceIndex': '1', 'numberOfGSVSentences': '3', 'numberOfSatellitesSeen': '11', 'azimuth_0': '111', 'azimuth_1': '270', 'azimuth_2': '010', 'azimuth_3': '292', 'elevation_0': '03', 'elevation_1': '15', 'elevation_2': '01', 'elevation_3': '06', 'satellitePRN_0': '03', 'satellitePRN_1': '04', 'satellitePRN_2': '06', 'satellitePRN_3': '13', 'signalToNoiseRatio_0': '00', 'signalToNoiseRatio_1': '00', 'signalToNoiseRatio_2': '00', 'signalToNoiseRatio_3': '00'}
        self._parserTest(GPGSV_FIRST, expected)

    def test_partialGSV(self) -> None:
        """
        A partial GSV sentence is correctly parsed.
        """
        expected = {'type': 'GPGSV', 'GSVSentenceIndex': '3', 'numberOfGSVSentences': '3', 'numberOfSatellitesSeen': '11', 'azimuth_0': '067', 'azimuth_1': '311', 'azimuth_2': '244', 'elevation_0': '42', 'elevation_1': '14', 'elevation_2': '05', 'satellitePRN_0': '22', 'satellitePRN_1': '24', 'satellitePRN_2': '27', 'signalToNoiseRatio_0': '42', 'signalToNoiseRatio_1': '43', 'signalToNoiseRatio_2': '00'}
        self._parserTest(GPGSV_LAST, expected)

    def test_fullHDT(self) -> None:
        """
        A full HDT sentence is correctly parsed.
        """
        expected = {'type': 'GPHDT', 'trueHeading': '038.005'}
        self._parserTest(GPHDT, expected)

    def test_typicalGSA(self) -> None:
        """
        A typical GSA sentence is correctly parsed.
        """
        expected = {'type': 'GPGSA', 'dataMode': 'A', 'fixType': '3', 'usedSatellitePRN_0': '19', 'usedSatellitePRN_1': '28', 'usedSatellitePRN_2': '14', 'usedSatellitePRN_3': '18', 'usedSatellitePRN_4': '27', 'usedSatellitePRN_5': '22', 'usedSatellitePRN_6': '31', 'usedSatellitePRN_7': '39', 'positionDilutionOfPrecision': '1.7', 'horizontalDilutionOfPrecision': '1.0', 'verticalDilutionOfPrecision': '1.3'}
        self._parserTest(GPGSA, expected)