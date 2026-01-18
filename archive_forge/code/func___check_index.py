import numbers
from . import _cluster  # type: ignore
def __check_index(index):
    if index is None:
        return np.zeros(1, dtype='intc')
    elif isinstance(index, numbers.Integral):
        return np.array([index], dtype='intc')
    elif isinstance(index, np.ndarray):
        return np.require(index, dtype='intc', requirements='C')
    else:
        return np.array(index, dtype='intc')