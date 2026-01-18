from .chemistry import equilibrium_quotient
def _get_rc_interval(stoich, c0):
    """get reaction coordinate interval"""
    limits = c0 / stoich
    if np.any(limits < 0):
        upper = -np.max(limits[np.argwhere(limits < 0)])
    else:
        upper = 0
    if np.any(limits > 0):
        lower = -np.min(limits[np.argwhere(limits > 0)])
    else:
        lower = 0
    if lower == 0 and upper == 0:
        raise ValueError('0-interval')
    else:
        return (lower, upper)