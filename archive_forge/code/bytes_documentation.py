from numba.core.extending import intrinsic
from llvmlite import ir
from numba.core import types, cgutils
Copy nbytes from *(src + src_offset) to *(dst + dst_offset)