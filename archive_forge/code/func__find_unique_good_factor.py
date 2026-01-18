from ..sage_helper import _within_sage, sage_method, SageNotAvailable
def _find_unique_good_factor(polynomial, eval_method):
    """
    Given a Sage polynomial, factor it. Return the unique factor for which the
    given eval_method returns an interval containing zero. If no or more than
    one factor have this property, raise an exception.
    """
    good_factors = [factor for factor, multiplicity in polynomial.factor() if 0 in eval_method(factor)]
    if not len(good_factors) == 1:
        raise _IsolateFactorError()
    return good_factors[0]