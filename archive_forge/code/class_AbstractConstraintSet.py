import sys
from pyasn1.type import error
class AbstractConstraintSet(AbstractConstraint):

    def __getitem__(self, idx):
        return self._values[idx]

    def __iter__(self):
        return iter(self._values)

    def __add__(self, value):
        return self.__class__(*self._values + (value,))

    def __radd__(self, value):
        return self.__class__(*(value,) + self._values)

    def __len__(self):
        return len(self._values)

    def _setValues(self, values):
        self._values = values
        for constraint in values:
            if constraint:
                self._valueMap.add(constraint)
                self._valueMap.update(constraint.getValueMap())