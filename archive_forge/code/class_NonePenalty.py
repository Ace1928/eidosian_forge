import numpy as np
class NonePenalty(Penalty):
    """
    A penalty that does not penalize.
    """

    def __init__(self, **kwds):
        super().__init__()
        if kwds:
            import warnings
            warnings.warn('keyword arguments are be ignored')

    def func(self, params):
        if params.ndim == 2:
            return np.zeros(params.shape[1:])
        else:
            return 0

    def deriv(self, params):
        return np.zeros(params.shape)

    def deriv2(self, params):
        return np.zeros(params.shape[0])