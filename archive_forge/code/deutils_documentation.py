from sympy.core import Pow
from sympy.core.function import Derivative, AppliedUndef
from sympy.core.relational import Equality
from sympy.core.symbol import Wild
This is a helper function to dsolve and pdsolve in the ode
    and pde modules.

    If the hint provided to the function is "default", then a dict with
    the following keys are returned

    'func'    - It provides the function for which the differential equation
                has to be solved. This is useful when the expression has
                more than one function in it.

    'default' - The default key as returned by classifier functions in ode
                and pde.py

    'hint'    - The hint given by the user for which the differential equation
                is to be solved. If the hint given by the user is 'default',
                then the value of 'hint' and 'default' is the same.

    'order'   - The order of the function as returned by ode_order

    'match'   - It returns the match as given by the classifier functions, for
                the default hint.

    If the hint provided to the function is not "default" and is not in
    ('all', 'all_Integral', 'best'), then a dict with the above mentioned keys
    is returned along with the keys which are returned when dict in
    classify_ode or classify_pde is set True

    If the hint given is in ('all', 'all_Integral', 'best'), then this function
    returns a nested dict, with the keys, being the set of classified hints
    returned by classifier functions, and the values being the dict of form
    as mentioned above.

    Key 'eq' is a common key to all the above mentioned hints which returns an
    expression if eq given by user is an Equality.

    See Also
    ========
    classify_ode(ode.py)
    classify_pde(pde.py)
    