from sympy.utilities.exceptions import sympy_deprecation_warning
from .facts import FactRules, FactKB
from .sympify import sympify
from sympy.core.random import _assumptions_shuffle as shuffle
from sympy.core.assumptions_generated import generated_assumptions as _assumptions
Precompute class level assumptions and generate handlers.

    This is called by Basic.__init_subclass__ each time a Basic subclass is
    defined.
    