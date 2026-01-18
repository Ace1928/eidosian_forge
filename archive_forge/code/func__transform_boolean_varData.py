from pyomo.common.collections import ComponentMap
from pyomo.common.errors import MouseTrap, DeveloperError
from pyomo.common.modeling import unique_component_name
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.core import (
from pyomo.core.base.block import _BlockData
from pyomo.core.base.boolean_var import _DeprecatedImplicitAssociatedBinaryVariable
from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr import (
from pyomo.core.expr.numvalue import native_logical_types, value
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.core.util import target_list
def _transform_boolean_varData(self, bool_vardata, new_varlists):
    parent_component = bool_vardata.parent_component()
    new_varlist = new_varlists.get(parent_component)
    if new_varlist is None and bool_vardata.get_associated_binary() is None:
        parent_block = bool_vardata.parent_block()
        new_var_list_name = unique_component_name(parent_block, parent_component.local_name + '_asbinary')
        new_varlist = VarList(domain=Binary)
        setattr(parent_block, new_var_list_name, new_varlist)
        new_varlists[parent_component] = new_varlist
    if bool_vardata.get_associated_binary() is None:
        new_binary_vardata = new_varlist.add()
        bool_vardata.associate_binary_var(new_binary_vardata)
        if bool_vardata.value is not None:
            new_binary_vardata.value = int(bool_vardata.value)
        if bool_vardata.fixed:
            new_binary_vardata.fix()