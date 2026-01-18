import sys
from . import model
from .error import VerificationError
from . import _imp_emulation as imp
def _generate_setup_custom(self):
    prnt = self._prnt
    prnt('static int _cffi_setup_custom(PyObject *lib)')
    prnt('{')
    prnt('  return %s;' % self._chained_list_constants[True])
    prnt('}')