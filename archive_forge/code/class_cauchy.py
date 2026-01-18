import numpy as np
import scipy.stats
import warnings
class cauchy(Cauchy):
    """
    The Cauchy (standard Cauchy CDF) transform

    .. deprecated: 0.14.0

       Use Cauchy instead.

    Notes
    -----
    cauchy is an alias of Cauchy.
    """

    def __init__(self):
        _link_deprecation_warning('cauchy', 'Cauchy')
        super().__init__()