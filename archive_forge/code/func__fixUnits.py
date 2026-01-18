import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
def _fixUnits(self, unitKey=None, valueKey=None, sourceKey=None, unit=None):
    """
        Fixes the units of a certain value. If the units are already
        acceptable (metric), does nothing.

        None of the keys are allowed to be the empty string.

        @param unit: The unit that is being converted I{from}. If unspecified
            or L{None}, asks the current sentence for the C{unitKey}. If that
            also fails, raises C{AttributeError}.
        @type unit: C{str}
        @param unitKey: The name of the key/attribute under which the unit can
            be found in the current sentence. If the C{unit} parameter is set,
            this parameter is not used.
        @type unitKey: C{str}
        @param sourceKey: The name of the key/attribute that contains the
            current value to be converted (expressed in units as defined
            according to the C{unit} parameter). If unset, will use the
            same key as the value key.
        @type sourceKey: C{str}
        @param valueKey: The key name in which the data will be stored in the
            C{_sentenceData} instance attribute. If unset, attempts to remove
            "Units" from the end of the C{unitKey} parameter. If that fails,
            raises C{ValueError}.
        @type valueKey: C{str}
        """
    if unit is None:
        unit = getattr(self.currentSentence, unitKey)
    if valueKey is None:
        if unitKey is not None and unitKey.endswith('Units'):
            valueKey = unitKey[:-5]
        else:
            raise ValueError("valueKey unspecified and couldn't be guessed")
    if sourceKey is None:
        sourceKey = valueKey
    if unit not in self._ACCEPTABLE_UNITS:
        converter = self._UNIT_CONVERTERS[unit]
        currentValue = getattr(self.currentSentence, sourceKey)
        self._sentenceData[valueKey] = converter(currentValue)