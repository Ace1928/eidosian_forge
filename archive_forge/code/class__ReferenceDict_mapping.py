from pyomo.common import DeveloperError
from pyomo.common.collections import (
from pyomo.common.modeling import NOTSET
from pyomo.core.base.set import DeclareGlobalSet, Set, SetOf, OrderedSetOf, _SetDataBase
from pyomo.core.base.component import Component, ComponentData
from pyomo.core.base.global_set import UnindexedComponent_set
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.indexed_component import IndexedComponent, normalize_index
from pyomo.core.base.indexed_component_slice import (
from pyomo.core.base.util import flatten_tuple
from pyomo.common.deprecation import deprecated
class _ReferenceDict_mapping(UserDict):

    def __init__(self, data):
        self.data = data