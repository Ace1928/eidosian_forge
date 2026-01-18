import os
from numpy import genfromtxt
class Y_arma02:

    def __init__(self, method='mle'):
        if method == 'mle':
            self.params = [0.169096401142, -0.683713393265]
            self.aic = 775.017701544762
            self.bic = 785.582084298349
            self.arroots = None
            self.maroots = [-1.092 + 0j, 1.3393 + 0j]
            self.bse = [0.049254112414, 0.050541821979]
            self.cov_params = [[0.002426, 0.00078704], [0.00078704, 0.0025545]]
            self.hqic = 779.269556450598
            self.llf = -384.508850772381
            self.resid = resids_mle[:, 5]
            self.fittedvalues = yhat_mle[:, 5]
            self.pvalues = [0.0006, 1.07e-41]
            self.tvalues = [3.433, -13.53]
            self.sigma2 = 1.122887152869 ** 2
        elif method == 'css':
            self.params = [0.175605240783, -0.688421349504]
            self.aic = 773.725350463014
            self.bic = 784.289733216601
            self.arroots = None
            self.maroots = [-1.0844 + 0j, 1.3395 + 0j]
            self.bse = [0.04850046, 0.05023068]
            self.cov_params = [[0.0023522942, 0.0007545702], [0.0007545702, 0.0025231209]]
            self.hqic = 777.97720536885
            self.llf = -383.862675231507
            self.resid = resids_css[:, 5]
            self.fittedvalues = yhat_css[:, 5]
            self.pvalues = [7.84e-05, 7.89e-53]
            self.tvalues = [3.620967, -13.705514]
            self.sigma2 = 1.123571177436 ** 2