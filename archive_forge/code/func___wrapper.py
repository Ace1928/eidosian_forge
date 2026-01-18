from typing import Any, Callable, Iterable, Optional, Set, TypeVar, Union
import warnings
import functools
from decorator import decorator
import numpy as np
from numpy.typing import DTypeLike
def __wrapper(function):
    vecfunc = np.vectorize(function, otypes=otypes, doc=doc, excluded=excluded, cache=cache, signature=signature)

    @functools.wraps(function)
    def _vec(*args, **kwargs):
        y = vecfunc(*args, **kwargs)
        if np.isscalar(args[0]):
            return y.item()
        else:
            return y
    return _vec