import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
def _fixTimestamp(self):
    """
        Turns the NMEAProtocol timestamp notation into a datetime.time object.
        The time in this object is expressed as Zulu time.
        """
    timestamp = self.currentSentence.timestamp.split('.')[0]
    timeObject = datetime.datetime.strptime(timestamp, '%H%M%S').time()
    self._sentenceData['_time'] = timeObject