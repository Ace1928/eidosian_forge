import os
from numpy import genfromtxt
class Y_arma11:

    def __init__(self, method='mle'):
        if method == 'mle':
            self.params = [0.788452102751, 0.381793815167]
            self.aic = 714.489820273473
            self.bic = 725.05420302706
            self.arroots = 1.2683 + 0j
            self.maroots = -2.6192 + 0j
            self.bse = [0.042075906061, 0.060925105865]
            self.hqic = 718.741675179309
            self.llf = -354.244910136737
            self.resid = resids_mle[:, 0]
            self.fittedvalues = yhat_mle[:, 0]
            self.pvalues = [2.39e-78, 3.69e-10]
            self.tvalues = [18.74, 6.267]
            self.sigma2 = 0.994743350844 ** 2
            self.cov_params = [[0.0017704, -0.0010612], [-0.0010612, 0.0037119]]
            self.forecast = forecast_results['fc11']
            self.forecasterr = forecast_results['fe11']
        elif method == 'css':
            self.params = [0.791515576984, 0.383078056824]
            self.aic = 710.99404717657
            self.bic = 721.546405865964
            self.arroots = [1.2634 + 0j]
            self.maroots = [-2.6104 + 0j]
            self.bse = [0.0424015620491, 0.0608752234378]
            self.cov_params = [[0.00179789246421, -0.0010619532154], [-0.0010619532154, 0.0037057928286]]
            self.hqic = 715.24154510855
            self.llf = -352.497023588285
            self.resid = resids_css[1:, 0]
            self.fittedvalues = yhat_css[1:, 0]
            self.pvalues = [7.02e-78, 5.53e-09]
            self.tvalues = [18.6671317239, 6.2928857557]
            self.sigma2 = 0.99671756278 ** 2