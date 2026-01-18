from cupy.cuda import runtime as _runtime
from cupyx.jit import _compile
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import BuiltinFunc as _BuiltinFunc
from cupyx.jit._internal_types import Constant as _Constant
from cupyx.jit._internal_types import Data as _Data
from cupyx.jit._internal_types import wraps_class_method as _wraps_class_method
def _check_include(env, header):
    flag = getattr(env.generated, f'include_{header}')
    if flag is False:
        env.generated.codes.append(_header_to_code[header])
        setattr(env.generated, f'include_{header}', True)