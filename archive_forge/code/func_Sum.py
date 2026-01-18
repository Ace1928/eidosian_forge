from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
from cupyx.jit import _internal_types
from cupy_backends.cuda.api import runtime as _runtime
@_internal_types.wraps_class_method
def Sum(self, env, instance, input) -> _internal_types.Data:
    if input.ctype != self.T:
        raise TypeError(f'Invalid input type {input.ctype}. ({self.T} is expected.)')
    return _internal_types.Data(f'{instance.code}.Sum({input.code})', input.ctype)