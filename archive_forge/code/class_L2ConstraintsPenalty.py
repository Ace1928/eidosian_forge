import numpy as np
class L2ConstraintsPenalty(ConstraintsPenalty):
    """convenience class of ConstraintsPenalty with L2 penalization
    """

    def __init__(self, weights=None, restriction=None, sigma_prior=None):
        if sigma_prior is not None:
            raise NotImplementedError('sigma_prior is not implemented yet')
        penalty = L2Univariate()
        super().__init__(penalty, weights=weights, restriction=restriction)