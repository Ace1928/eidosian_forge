from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
from cupyx.jit import _internal_types
from cupy_backends.cuda.api import runtime as _runtime
class _TempStorageType(_cuda_types.TypeBase):

    def __init__(self, parent_type):
        assert isinstance(parent_type, _CubReduceBaseType)
        self.parent_type = parent_type
        super().__init__()

    def __str__(self) -> str:
        return f'typename {self.parent_type}::TempStorage'