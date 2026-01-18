import hashlib
import os
import tempfile
from ..common import _build
from ..common.backend import get_cuda_version_key
from ..common.build import is_hip
from ..runtime.cache import get_cache_manager
from .utils import generate_cu_signature
def _extracted_type(ty):
    if ty[0] == '*':
        return 'PyObject*'
    return {'i1': 'int32_t', 'i32': 'int32_t', 'i64': 'int64_t', 'u32': 'uint32_t', 'u64': 'uint64_t', 'fp16': 'float', 'bf16': 'float', 'fp32': 'float', 'f32': 'float', 'fp64': 'double'}[ty]