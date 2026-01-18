from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
from cupyx.jit import _internal_types
from cupy_backends.cuda.api import runtime as _runtime
def _get_cub_namespace():
    return 'hipcub' if _runtime.is_hip else 'cub'