import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
def _fixGSV(self):
    """
        Parses partial visible satellite information from a GSV sentence.
        """
    beaconInformation = base.BeaconInformation()
    self._sentenceData['_partialBeaconInformation'] = beaconInformation
    keys = ('satellitePRN', 'azimuth', 'elevation', 'signalToNoiseRatio')
    for index in range(4):
        prn, azimuth, elevation, snr = (getattr(self.currentSentence, attr) for attr in ('%s_%i' % (key, index) for key in keys))
        if prn is None or snr is None:
            continue
        satellite = base.Satellite(prn, azimuth, elevation, snr)
        beaconInformation.seenBeacons.add(satellite)