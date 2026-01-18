import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
def _sentenceSpecificFix(self):
    """
        Executes a fix for a specific type of sentence.
        """
    fixer = self._SPECIFIC_SENTENCE_FIXES.get(self.currentSentence.type)
    if fixer is not None:
        fixer(self)