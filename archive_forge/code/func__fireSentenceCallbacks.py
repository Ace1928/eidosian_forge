import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
def _fireSentenceCallbacks(self):
    """
        Fires sentence callbacks for the current sentence.

        A callback will only fire if all of the keys it requires are present
        in the current state and at least one such field was altered in the
        current sentence.

        The callbacks will only be fired with data from L{_state}.
        """
    iface = ipositioning.IPositioningReceiver
    for name, method in iface.namesAndDescriptions():
        callback = getattr(self._receiver, name)
        kwargs = {}
        atLeastOnePresentInSentence = False
        try:
            for field in method.positional:
                if field in self._sentenceData:
                    atLeastOnePresentInSentence = True
                kwargs[field] = self._state[field]
        except KeyError:
            continue
        if atLeastOnePresentInSentence:
            callback(**kwargs)