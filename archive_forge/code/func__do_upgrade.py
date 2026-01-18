import numpy as np
import numpy.core.numeric as nx
from numpy.compat import asbytes, asunicode
def _do_upgrade(self):
    if self._locked:
        errmsg = 'Converter is locked and cannot be upgraded'
        raise ConverterLockError(errmsg)
    _statusmax = len(self._mapper)
    _status = self._status
    if _status == _statusmax:
        errmsg = 'Could not find a valid conversion function'
        raise ConverterError(errmsg)
    elif _status < _statusmax - 1:
        _status += 1
    self.type, self.func, default = self._mapper[_status]
    self._status = _status
    if self._initial_default is not None:
        self.default = self._initial_default
    else:
        self.default = default