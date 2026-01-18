import sys
from pyasn1.type import error
def _testValue(self, value, idx):
    for constraint in self._values:
        try:
            constraint(value, idx)
        except error.ValueConstraintError:
            pass
        else:
            return
    raise error.ValueConstraintError('all of %s failed for "%s"' % (self._values, value))