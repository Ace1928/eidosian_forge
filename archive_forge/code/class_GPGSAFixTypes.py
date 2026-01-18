import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
class GPGSAFixTypes(Values):
    """
    The possible fix types of a GPGSA sentence.

    @cvar GSA_NO_FIX: The sentence reports no fix at all.
    @cvar GSA_2D_FIX: The sentence reports a 2D fix: position but no altitude.
    @cvar GSA_3D_FIX: The sentence reports a 3D fix: position with altitude.
    """
    GSA_NO_FIX = ValueConstant('1')
    GSA_2D_FIX = ValueConstant('2')
    GSA_3D_FIX = ValueConstant('3')