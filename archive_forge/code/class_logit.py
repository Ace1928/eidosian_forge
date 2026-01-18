import numpy as np
import scipy.stats
import warnings
class logit(Logit):
    """
    Alias of Logit

    .. deprecated: 0.14.0

       Use Logit instead.
    """

    def __init__(self):
        _link_deprecation_warning('logit', 'Logit')
        super().__init__()