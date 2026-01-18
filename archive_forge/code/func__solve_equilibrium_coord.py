from .chemistry import equilibrium_quotient
def _solve_equilibrium_coord(c0, stoich, K, activity_product=None):
    from scipy.optimize import brentq
    mask, = np.nonzero(stoich)
    stoich_m = stoich[mask]
    c0_m = c0[mask]
    lower, upper = _get_rc_interval(stoich_m, c0_m)
    return brentq(equilibrium_residual, lower, upper, (c0_m, stoich_m, K, activity_product))