import logging
import sys
from copy import deepcopy
from pickle import PickleError
from weakref import ref as weakref_ref
import pyomo.common
from pyomo.common import DeveloperError
from pyomo.common.autoslots import AutoSlots, fast_deepcopy
from pyomo.common.collections import OrderedDict
from pyomo.common.deprecation import (
from pyomo.common.factory import Factory
from pyomo.common.formatting import tabular_writer, StreamIndenter
from pyomo.common.modeling import NOTSET
from pyomo.common.sorting import sorted_robust
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.base.component_namer import name_repr, index_repr
from pyomo.core.base.global_set import UnindexedComponent_index
def _deepcopy_field(self, memo, slot_name, value):
    saved_memo = len(memo)
    try:
        return fast_deepcopy(value, memo)
    except CloneError:
        raise
    except:
        for _ in range(len(memo) - saved_memo):
            memo.popitem()
        if '__block_scope__' not in memo:
            logger.warning("\n                    Uncopyable field encountered when deep\n                    copying outside the scope of Block.clone().\n                    There is a distinct possibility that the new\n                    copy is not complete.  To avoid this\n                    situation, either use Block.clone() or set\n                    'paranoid' mode by adding '__paranoid__' ==\n                    True to the memo before calling\n                    copy.deepcopy.")
        if self.model() is self:
            what = 'Model'
        else:
            what = 'Component'
        logger.error("Unable to clone Pyomo component attribute.\n%s '%s' contains an uncopyable field '%s' (%s).  Setting field to `None` on new object" % (what, self.name, slot_name, type(value)))
        if not self.parent_component()._constructed:
            raise CloneError('Uncopyable attribute (%s) encountered when cloning component %s on an abstract block.  The resulting instance is therefore missing data from the original abstract model and likely will not construct correctly.  Consider changing how you initialize this component or using a ConcreteModel.' % (slot_name, self.name))
    return None