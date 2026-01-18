from numpy.core import umath as um
def _disables_array_ufunc(obj):
    """True when __array_ufunc__ is set to None."""
    try:
        return obj.__array_ufunc__ is None
    except AttributeError:
        return False