import operator
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import types
from numba.core.extending import overload_method

    Return an enum member by index name.
    