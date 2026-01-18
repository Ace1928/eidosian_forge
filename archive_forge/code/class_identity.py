import numpy as np
import scipy.stats
import warnings
class identity(Identity):
    """
    Deprecated alias of Identity.

    .. deprecated: 0.14.0

        Use Identity instead.
    """

    def __init__(self):
        _link_deprecation_warning('identity', 'Identity')
        super().__init__()