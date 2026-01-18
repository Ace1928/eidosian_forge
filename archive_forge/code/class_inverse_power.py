import numpy as np
import scipy.stats
import warnings
class inverse_power(InversePower):
    """
    Deprecated alias of InversePower.

    .. deprecated: 0.14.0

        Use InversePower instead.
    """

    def __init__(self):
        _link_deprecation_warning('inverse_power', 'InversePower')
        super().__init__()