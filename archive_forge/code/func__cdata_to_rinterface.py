from typing import Callable
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import _rinterface_capi as _rinterface
def _cdata_to_rinterface(cdata):
    scaps = _rinterface.SexpCapsule(cdata)
    t = cdata.sxpinfo.type
    if t in _R_RPY2_MAP:
        cls = _R_RPY2_MAP[t]
    else:
        cls = _R_RPY2_DEFAULT_MAP
    return cls(scaps)