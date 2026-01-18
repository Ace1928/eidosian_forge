import enum
import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.config import In
from pyomo.common.deprecation import deprecated
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.pyomo_typing import overload
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.component import ActiveComponent, ModelComponentFactory
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import Initializer
class SuffixDirection(enum.IntEnum):
    """Suffix data flow definition.

    This identifies if the specific Suffix is to be sent to the solver,
    read from the solver output, both, or neither:

    - LOCAL: Suffix is local to Pyomo and should not be sent to or read
      from the solver.

    - EXPORT: Suffix should be sent to the solver as supplemental model
      information.

    - IMPORT: Suffix values will be returned from the solver and should
      be read from the solver output.

    - IMPORT_EXPORT: The Suffix is both an EXPORT and IMPORT suffix.

    """
    LOCAL = 0
    EXPORT = 1
    IMPORT = 2
    IMPORT_EXPORT = 3