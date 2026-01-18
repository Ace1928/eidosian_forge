import numpy as np
import scipy.stats
import warnings
class loglog(LogLog):
    """
    The LogLog transform link function.

    .. deprecated: 0.14.0

       Use LogLog instead.

    Notes
    -----
    g(`p`) = -log(-log(`p`))

    loglog is an alias for LogLog
    loglog = LogLog()
    """

    def __init__(self):
        _link_deprecation_warning('loglog', 'LogLog')
        super().__init__()