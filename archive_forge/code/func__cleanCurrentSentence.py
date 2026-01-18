import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
def _cleanCurrentSentence(self):
    """
        Cleans the current sentence.
        """
    for key in sorted(self.currentSentence.presentAttributes):
        fixer = self._FIXERS.get(key, None)
        if fixer is not None:
            fixer(self)