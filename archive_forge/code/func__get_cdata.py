from typing import Callable
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import _rinterface_capi as _rinterface
def _get_cdata(obj):
    cls = _PY_R_MAP.get(type(obj))
    if cls is False:
        cdata = obj
    elif cls is None:
        try:
            cdata = obj.__sexp__._cdata
        except AttributeError:
            raise ValueError('Not an rpy2 R object and unable to map it to one: %s' % repr(obj))
    else:
        cdata = cls(obj)
    return cdata