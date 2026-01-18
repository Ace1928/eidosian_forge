from numba.core import types, errors, ir, sigutils, ir_utils
from numba.core.typing.typeof import typeof_impl
from numba.core.transforms import find_region_inout_vars
from numba.core.ir_utils import build_definitions
import numba
def _get_var_parent(name):
    """Get parent of the variable given its name
    """
    if not name.startswith('$'):
        return name.split('.')[0]