import warnings
from .._utils import set_module
from ._machar import MachAr
from . import numeric
from . import numerictypes as ntypes
from .numeric import array, inf, NaN
from .umath import log10, exp2, nextafter, isnan
def _get_machar(ftype):
    """ Get MachAr instance or MachAr-like instance

    Get parameters for floating point type, by first trying signatures of
    various known floating point types, then, if none match, attempting to
    identify parameters by analysis.

    Parameters
    ----------
    ftype : class
        Numpy floating point type class (e.g. ``np.float64``)

    Returns
    -------
    ma_like : instance of :class:`MachAr` or :class:`MachArLike`
        Object giving floating point parameters for `ftype`.

    Warns
    -----
    UserWarning
        If the binary signature of the float type is not in the dictionary of
        known float types.
    """
    params = _MACHAR_PARAMS.get(ftype)
    if params is None:
        raise ValueError(repr(ftype))
    key = (ftype(-1.0) / ftype(10.0)).newbyteorder('<').tobytes()
    ma_like = None
    if ftype == ntypes.longdouble:
        ma_like = _KNOWN_TYPES.get(key[:10])
    if ma_like is None:
        ma_like = _KNOWN_TYPES.get(key)
    if ma_like is None and len(key) == 16:
        _kt = {k[:10]: v for k, v in _KNOWN_TYPES.items() if len(k) == 16}
        ma_like = _kt.get(key[:10])
    if ma_like is not None:
        return ma_like
    warnings.warn(f'Signature {key} for {ftype} does not match any known type: falling back to type probe function.\nThis warnings indicates broken support for the dtype!', UserWarning, stacklevel=2)
    return _discovered_machar(ftype)