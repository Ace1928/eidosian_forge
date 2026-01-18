import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
def _validateCurrentSentence(self):
    """
        Tests if a sentence contains a valid fix.
        """
    if self.currentSentence.fixQuality is GPGGAFixQualities.INVALID_FIX or self.currentSentence.dataMode is GPGLLGPRMCFixQualities.VOID or self.currentSentence.fixType is GPGSAFixTypes.GSA_NO_FIX:
        raise base.InvalidSentence('bad sentence')