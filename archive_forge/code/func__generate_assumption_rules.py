from sympy.utilities.exceptions import sympy_deprecation_warning
from .facts import FactRules, FactKB
from .sympify import sympify
from sympy.core.random import _assumptions_shuffle as shuffle
from sympy.core.assumptions_generated import generated_assumptions as _assumptions
def _generate_assumption_rules():
    """ Generate the default assumption rules

    This method should only be called to update the pre-generated
    assumption rules.

    To update the pre-generated assumptions run: bin/ask_update.py

    """
    _assume_rules = FactRules(['integer        ->  rational', 'rational       ->  real', 'rational       ->  algebraic', 'algebraic      ->  complex', 'transcendental ==  complex & !algebraic', 'real           ->  hermitian', 'imaginary      ->  complex', 'imaginary      ->  antihermitian', 'extended_real  ->  commutative', 'complex        ->  commutative', 'complex        ->  finite', 'odd            ==  integer & !even', 'even           ==  integer & !odd', 'real           ->  complex', 'extended_real  ->  real | infinite', 'real           ==  extended_real & finite', 'extended_real        ==  extended_negative | zero | extended_positive', 'extended_negative    ==  extended_nonpositive & extended_nonzero', 'extended_positive    ==  extended_nonnegative & extended_nonzero', 'extended_nonpositive ==  extended_real & !extended_positive', 'extended_nonnegative ==  extended_real & !extended_negative', 'real           ==  negative | zero | positive', 'negative       ==  nonpositive & nonzero', 'positive       ==  nonnegative & nonzero', 'nonpositive    ==  real & !positive', 'nonnegative    ==  real & !negative', 'positive       ==  extended_positive & finite', 'negative       ==  extended_negative & finite', 'nonpositive    ==  extended_nonpositive & finite', 'nonnegative    ==  extended_nonnegative & finite', 'nonzero        ==  extended_nonzero & finite', 'zero           ->  even & finite', 'zero           ==  extended_nonnegative & extended_nonpositive', 'zero           ==  nonnegative & nonpositive', 'nonzero        ->  real', 'prime          ->  integer & positive', 'composite      ->  integer & positive & !prime', '!composite     ->  !positive | !even | prime', 'irrational     ==  real & !rational', 'imaginary      ->  !extended_real', 'infinite       ==  !finite', 'noninteger     ==  extended_real & !integer', 'extended_nonzero == extended_real & !zero'])
    return _assume_rules