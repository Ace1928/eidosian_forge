import numpy as np
import scipy.stats
import warnings
class logc(LogC):
    """
    The log-complement transform

    .. deprecated: 0.14.0

       Use LogC instead.

    Notes
    -----
    logc is a an alias of LogC.
    """

    def __init__(self):
        _link_deprecation_warning('logc', 'LogC')
        super().__init__()