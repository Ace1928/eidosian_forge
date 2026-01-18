import warnings
from .._utils import set_module
from ._machar import MachAr
from . import numeric
from . import numerictypes as ntypes
from .numeric import array, inf, NaN
from .umath import log10, exp2, nextafter, isnan
def _discovered_machar(ftype):
    """ Create MachAr instance with found information on float types

    TODO: MachAr should be retired completely ideally.  We currently only
          ever use it system with broken longdouble (valgrind, WSL).
    """
    params = _MACHAR_PARAMS[ftype]
    return MachAr(lambda v: array([v], ftype), lambda v: _fr0(v.astype(params['itype']))[0], lambda v: array(_fr0(v)[0], ftype), lambda v: params['fmt'] % array(_fr0(v)[0], ftype), params['title'])