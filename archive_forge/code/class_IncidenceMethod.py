import enum
from pyomo.common.config import ConfigDict, ConfigValue, InEnum
from pyomo.common.modeling import NOTSET
from pyomo.repn.plugins.nl_writer import AMPLRepnVisitor, text_nl_template
from pyomo.repn.util import FileDeterminism, FileDeterminism_to_SortComponents
class IncidenceMethod(enum.Enum):
    """Methods for identifying variables that participate in expressions"""
    identify_variables = 0
    'Use ``pyomo.core.expr.visitor.identify_variables``'
    standard_repn = 1
    'Use ``pyomo.repn.standard_repn.generate_standard_repn``'
    standard_repn_compute_values = 2
    'Use ``pyomo.repn.standard_repn.generate_standard_repn`` with\n    ``compute_values=True``\n    '
    ampl_repn = 3
    'Use ``pyomo.repn.plugins.nl_writer.AMPLRepnVisitor``'