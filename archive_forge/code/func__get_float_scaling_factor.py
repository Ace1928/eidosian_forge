from pyomo.common.collections import ComponentMap
from pyomo.core.base import (
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.core.base import TransformationFactory
from pyomo.core.base.suffix import SuffixFinder
from pyomo.core.expr import replace_expressions
from pyomo.util.components import rename_components
def _get_float_scaling_factor(self, component):
    if self._suffix_finder is None:
        self._suffix_finder = SuffixFinder('scaling_factor', 1.0)
    return self._suffix_finder.find(component)