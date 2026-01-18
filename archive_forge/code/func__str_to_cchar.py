from typing import Callable
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import _rinterface_capi as _rinterface
def _str_to_cchar(s: str, encoding: str='utf-8'):
    b = s.encode(encoding)
    return ffi.new('char[]', b)