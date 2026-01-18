from itertools import cycle
import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import check_nD
class _fcycle:

    def __init__(self, iterable):
        """Call functions from the iterable each time it is called."""
        self.funcs = cycle(iterable)

    def __call__(self, *args, **kwargs):
        f = next(self.funcs)
        return f(*args, **kwargs)