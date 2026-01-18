import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
Convert bytes array item or bytes instance to UTF-8 str.

    Note: The usage of _to_str method can be eliminated once all
    Python bytes operations are implemented for numba Bytes objects.
    