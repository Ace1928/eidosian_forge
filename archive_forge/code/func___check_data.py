import numbers
from . import _cluster  # type: ignore
def __check_data(data):
    if isinstance(data, np.ndarray):
        data = np.require(data, dtype='d', requirements='C')
    else:
        data = np.array(data, dtype='d')
    if data.ndim != 2:
        raise ValueError('data should be 2-dimensional')
    if np.isnan(data).any():
        raise ValueError('data contains NaN values')
    return data