import numpy as np
import scipy.stats
import warnings
class sqrt(Sqrt):
    """
    Deprecated alias of Sqrt.

    .. deprecated: 0.14.0

        Use Sqrt instead.
    """

    def __init__(self):
        _link_deprecation_warning('sqrt', 'Sqrt')
        super().__init__()