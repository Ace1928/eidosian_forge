import sys
from pyasn1.type import error
def _setValues(self, values):
    self._values = values
    for constraint in values:
        if constraint:
            self._valueMap.add(constraint)
            self._valueMap.update(constraint.getValueMap())