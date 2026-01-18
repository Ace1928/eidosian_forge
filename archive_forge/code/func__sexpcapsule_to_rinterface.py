from typing import Callable
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import _rinterface_capi as _rinterface
def _sexpcapsule_to_rinterface(scaps: '_rinterface.SexpCapsule'):
    cls = _R_RPY2_MAP.get(scaps.typeof, _R_RPY2_DEFAULT_MAP)
    return cls(scaps)