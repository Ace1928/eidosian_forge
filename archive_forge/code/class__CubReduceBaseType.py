from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
from cupyx.jit import _internal_types
from cupy_backends.cuda.api import runtime as _runtime
class _CubReduceBaseType(_cuda_types.TypeBase):

    def _instantiate(self, env, temp_storage) -> _internal_types.Data:
        _include_cub(env)
        if temp_storage.ctype != self.TempStorage:
            raise TypeError(f'Invalid temp_storage type {temp_storage.ctype}. ({self.TempStorage} is expected.)')
        return _internal_types.Data(f'{self}({temp_storage.code})', self)

    @_internal_types.wraps_class_method
    def Sum(self, env, instance, input) -> _internal_types.Data:
        if input.ctype != self.T:
            raise TypeError(f'Invalid input type {input.ctype}. ({self.T} is expected.)')
        return _internal_types.Data(f'{instance.code}.Sum({input.code})', input.ctype)

    @_internal_types.wraps_class_method
    def Reduce(self, env, instance, input, reduction_op):
        if input.ctype != self.T:
            raise TypeError(f'Invalid input type {input.ctype}. ({self.T} is expected.)')
        return _internal_types.Data(f'{instance.code}.Reduce({input.code}, {reduction_op.code})', input.ctype)