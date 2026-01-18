from sympy.core import Pow
from sympy.core.function import Derivative, AppliedUndef
from sympy.core.relational import Equality
from sympy.core.symbol import Wild
def _desolve(eq, func=None, hint='default', ics=None, simplify=True, *, prep=True, **kwargs):
    """This is a helper function to dsolve and pdsolve in the ode
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
    """
    if isinstance(eq, Equality):
        eq = eq.lhs - eq.rhs
    if prep or func is None:
        eq, func = _preprocess(eq, func)
        prep = False
    type = kwargs.get('type', None)
    xi = kwargs.get('xi')
    eta = kwargs.get('eta')
    x0 = kwargs.get('x0', 0)
    terms = kwargs.get('n')
    if type == 'ode':
        from sympy.solvers.ode import classify_ode, allhints
        classifier = classify_ode
        string = 'ODE '
        dummy = ''
    elif type == 'pde':
        from sympy.solvers.pde import classify_pde, allhints
        classifier = classify_pde
        string = 'PDE '
        dummy = 'p'
    if kwargs.get('classify', True):
        hints = classifier(eq, func, dict=True, ics=ics, xi=xi, eta=eta, n=terms, x0=x0, hint=hint, prep=prep)
    else:
        hints = kwargs.get('hint', {'default': hint, hint: kwargs['match'], 'order': kwargs['order']})
    if not hints['default']:
        if hint not in allhints and hint != 'default':
            raise ValueError('Hint not recognized: ' + hint)
        elif hint not in hints['ordered_hints'] and hint != 'default':
            raise ValueError(string + str(eq) + ' does not match hint ' + hint)
        elif hints['order'] == 0:
            raise ValueError(str(eq) + ' is not a solvable differential equation in ' + str(func))
        else:
            raise NotImplementedError(dummy + 'solve' + ': Cannot solve ' + str(eq))
    if hint == 'default':
        return _desolve(eq, func, ics=ics, hint=hints['default'], simplify=simplify, prep=prep, x0=x0, classify=False, order=hints['order'], match=hints[hints['default']], xi=xi, eta=eta, n=terms, type=type)
    elif hint in ('all', 'all_Integral', 'best'):
        retdict = {}
        gethints = set(hints) - {'order', 'default', 'ordered_hints'}
        if hint == 'all_Integral':
            for i in hints:
                if i.endswith('_Integral'):
                    gethints.remove(i[:-len('_Integral')])
            for k in ['1st_homogeneous_coeff_best', '1st_power_series', 'lie_group', '2nd_power_series_ordinary', '2nd_power_series_regular']:
                if k in gethints:
                    gethints.remove(k)
        for i in gethints:
            sol = _desolve(eq, func, ics=ics, hint=i, x0=x0, simplify=simplify, prep=prep, classify=False, n=terms, order=hints['order'], match=hints[i], type=type)
            retdict[i] = sol
        retdict['all'] = True
        retdict['eq'] = eq
        return retdict
    elif hint not in allhints:
        raise ValueError('Hint not recognized: ' + hint)
    elif hint not in hints:
        raise ValueError(string + str(eq) + ' does not match hint ' + hint)
    else:
        hints['hint'] = hint
    hints.update({'func': func, 'eq': eq})
    return hints