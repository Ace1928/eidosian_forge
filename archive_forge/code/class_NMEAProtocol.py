import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
class NMEAProtocol(LineReceiver, _sentence._PositioningSentenceProducerMixin):
    """
    A protocol that parses and verifies the checksum of an NMEA sentence (in
    string form, not L{NMEASentence}), and delegates to a receiver.

    It receives lines and verifies these lines are NMEA sentences. If
    they are, verifies their checksum and unpacks them into their
    components. It then wraps them in L{NMEASentence} objects and
    calls the appropriate receiver method with them.

    @cvar _SENTENCE_CONTENTS: Has the field names in an NMEA sentence for each
        sentence type (in order, obviously).
    @type _SENTENCE_CONTENTS: C{dict} of bytestrings to C{list}s of C{str}
    @param receiver: A receiver for NMEAProtocol sentence objects.
    @type receiver: L{INMEAReceiver}
    @param sentenceCallback: A function that will be called with a new
        L{NMEASentence} when it is created. Useful for massaging data from
        particularly misbehaving NMEA receivers.
    @type sentenceCallback: unary callable
    """

    def __init__(self, receiver, sentenceCallback=None):
        """
        Initializes an NMEAProtocol.

        @param receiver: A receiver for NMEAProtocol sentence objects.
        @type receiver: L{INMEAReceiver}
        @param sentenceCallback: A function that will be called with a new
            L{NMEASentence} when it is created. Useful for massaging data from
            particularly misbehaving NMEA receivers.
        @type sentenceCallback: unary callable
        """
        self._receiver = receiver
        self._sentenceCallback = sentenceCallback

    def lineReceived(self, rawSentence):
        """
        Parses the data from the sentence and validates the checksum.

        @param rawSentence: The NMEA positioning sentence.
        @type rawSentence: C{bytes}
        """
        sentence = rawSentence.strip()
        _validateChecksum(sentence)
        splitSentence = _split(sentence)
        sentenceType = nativeString(splitSentence[0])
        contents = [nativeString(x) for x in splitSentence[1:]]
        try:
            keys = self._SENTENCE_CONTENTS[sentenceType]
        except KeyError:
            raise ValueError('unknown sentence type %s' % sentenceType)
        sentenceData = {'type': sentenceType}
        for key, value in zip(keys, contents):
            if key is not None and value != '':
                sentenceData[key] = value
        sentence = NMEASentence(sentenceData)
        if self._sentenceCallback is not None:
            self._sentenceCallback(sentence)
        self._receiver.sentenceReceived(sentence)
    _SENTENCE_CONTENTS = {'GPGGA': ['timestamp', 'latitudeFloat', 'latitudeHemisphere', 'longitudeFloat', 'longitudeHemisphere', 'fixQuality', 'numberOfSatellitesSeen', 'horizontalDilutionOfPrecision', 'altitude', 'altitudeUnits', 'heightOfGeoidAboveWGS84', 'heightOfGeoidAboveWGS84Units', None, None], 'GPRMC': ['timestamp', 'dataMode', 'latitudeFloat', 'latitudeHemisphere', 'longitudeFloat', 'longitudeHemisphere', 'speedInKnots', 'trueHeading', 'datestamp', 'magneticVariation', 'magneticVariationDirection'], 'GPGSV': ['numberOfGSVSentences', 'GSVSentenceIndex', 'numberOfSatellitesSeen', 'satellitePRN_0', 'elevation_0', 'azimuth_0', 'signalToNoiseRatio_0', 'satellitePRN_1', 'elevation_1', 'azimuth_1', 'signalToNoiseRatio_1', 'satellitePRN_2', 'elevation_2', 'azimuth_2', 'signalToNoiseRatio_2', 'satellitePRN_3', 'elevation_3', 'azimuth_3', 'signalToNoiseRatio_3'], 'GPGLL': ['latitudeFloat', 'latitudeHemisphere', 'longitudeFloat', 'longitudeHemisphere', 'timestamp', 'dataMode'], 'GPHDT': ['trueHeading'], 'GPTRF': ['datestamp', 'timestamp', 'latitudeFloat', 'latitudeHemisphere', 'longitudeFloat', 'longitudeHemisphere', 'elevation', 'numberOfIterations', 'numberOfDopplerIntervals', 'updateDistanceInNauticalMiles', 'satellitePRN'], 'GPGSA': ['dataMode', 'fixType', 'usedSatellitePRN_0', 'usedSatellitePRN_1', 'usedSatellitePRN_2', 'usedSatellitePRN_3', 'usedSatellitePRN_4', 'usedSatellitePRN_5', 'usedSatellitePRN_6', 'usedSatellitePRN_7', 'usedSatellitePRN_8', 'usedSatellitePRN_9', 'usedSatellitePRN_10', 'usedSatellitePRN_11', 'positionDilutionOfPrecision', 'horizontalDilutionOfPrecision', 'verticalDilutionOfPrecision']}