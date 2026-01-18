import numpy as np
import scipy.stats
import warnings
class InverseSquared(Power):
    """
    The inverse squared transform

    Notes
    -----
    g(`p`) = 1/(`p`\\*\\*2)

    Alias of statsmodels.family.links.Power(power=2.)
    """

    def __init__(self):
        super().__init__(power=-2.0)