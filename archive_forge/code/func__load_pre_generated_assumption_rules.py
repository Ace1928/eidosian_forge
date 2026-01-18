from sympy.utilities.exceptions import sympy_deprecation_warning
from .facts import FactRules, FactKB
from .sympify import sympify
from sympy.core.random import _assumptions_shuffle as shuffle
from sympy.core.assumptions_generated import generated_assumptions as _assumptions
def _load_pre_generated_assumption_rules():
    """ Load the assumption rules from pre-generated data

    To update the pre-generated data, see :method::`_generate_assumption_rules`
    """
    _assume_rules = FactRules._from_python(_assumptions)
    return _assume_rules