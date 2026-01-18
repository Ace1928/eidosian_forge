import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
def _isLastGSVSentence(self):
    """
        Tests if this current GSV sentence is the final one in a sequence.

        @return: C{True} if this is the last GSV sentence.
        @rtype: C{bool}
        """
    return self.GSVSentenceIndex == self.numberOfGSVSentences