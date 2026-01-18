import numbers
from . import _cluster  # type: ignore
def __check_weight(weight, ndata):
    if weight is None:
        return np.ones(ndata, dtype='d')
    if isinstance(weight, np.ndarray):
        weight = np.require(weight, dtype='d', requirements='C')
    else:
        weight = np.array(weight, dtype='d')
    if np.isnan(weight).any():
        raise ValueError('weight contains NaN values')
    return weight