import inspect
from pyomo.common.deprecation import deprecation_warning
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.transformation import (
from pyomo.scripting.interface import (
class IPyomoExpression(DeprecatedInterface):

    def type(self):
        """Return the type of expression"""

    def create(self, args):
        """Create an instance of this expression type"""