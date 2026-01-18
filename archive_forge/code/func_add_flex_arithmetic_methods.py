from __future__ import annotations
import operator
from pandas.core.dtypes.generic import (
from pandas.core.ops import roperator
def add_flex_arithmetic_methods(cls) -> None:
    """
    Adds the full suite of flex arithmetic methods (``pow``, ``mul``, ``add``)
    to the class.

    Parameters
    ----------
    cls : class
        flex methods will be defined and pinned to this class
    """
    flex_arith_method, flex_comp_method = _get_method_wrappers(cls)
    new_methods = _create_methods(cls, flex_arith_method, flex_comp_method)
    new_methods.update({'multiply': new_methods['mul'], 'subtract': new_methods['sub'], 'divide': new_methods['div']})
    assert not any((kname in new_methods for kname in ('ror_', 'rxor', 'rand_')))
    _add_methods(cls, new_methods=new_methods)