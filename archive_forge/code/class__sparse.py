from pyomo.common.deprecation import deprecated
from pyomo.core import (
from pyomo.repn import generate_standard_repn
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.core.plugins.transform.standard_form import StandardForm
from pyomo.core.plugins.transform.util import partial, process_canonical_repn
class _sparse(dict):
    """
    Represents a sparse map. Uses a user-provided value to initialize
    entries. If the default value is a callable object, it is called
    with no arguments.

    Examples

      # Sparse vector
      v = _sparse(0)

      # 2-dimensional sparse matrix
      A = _sparse(lambda: _sparse(0))

    """

    def __init__(self, default, *args, **kwds):
        dict.__init__(self, *args, **kwds)
        if hasattr(default, '__call__'):
            self._default_value = None
            self._default_func = default
        else:
            self._default_value = default
            self._default_func = None

    def __getitem__(self, ndx):
        if ndx in self:
            return dict.__getitem__(self, ndx)
        elif self._default_func is not None:
            return self._default_func()
        else:
            return self._default_value