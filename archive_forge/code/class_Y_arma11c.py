import os
from numpy import genfromtxt
class Y_arma11c:

    def __init__(self, method='mle'):
        if method == 'mle':
            self.params = [4.85647575943, 0.664363281011, 0.407547531124]
            self.aic = 737.922644877973
            self.bic = 752.008488549422
            self.arroots = [1.5052 + 0j]
            self.maroots = [-2.4537 + 0j]
            self.bse = [0.27316417696, 0.055495689209, 0.068249092654]
            self.cov_params = [[0.074619, -0.00012834, 1.5413e-05], [-0.00012834, 0.0030798, -0.0020242], [1.5413e-05, -0.0020242, 0.0046579]]
            self.hqic = 743.591784752421
            self.llf = -364.961322438987
            self.resid = residsc_mle[:, 0]
            self.fittedvalues = yhatc_mle[:, 0]
            self.pvalues = [1.04e-70, 5.02e-33, 2.35e-09]
            self.tvalues = [17.78, 11.97, 5.971]
            self.sigma2 = 1.039168068701 ** 2
            self.forecast = forecast_results['fc11c']
            self.forecasterr = forecast_results['fe11c']
        elif method == 'css':
            self.params = [4.872477127267, 0.666395534262, 0.409517026658]
            self.aic = 734.613526514951
            self.bic = 748.68333810081
            self.arroots = [1.5006 + 0j]
            self.maroots = [-2.4419 + 0.0]
            self.bse = [0.2777238133284, 0.0557583459688, 0.0681432545482]
            self.cov_params = [[0.0771305164897, 5.65375305967e-06, 1.29481824075e-06], [5.65375305967e-06, 0.00310899314518, -0.00202754322743], [1.29481824075e-06, -0.00202754322743, 0.00464350314042]]
            self.hqic = 740.276857090925
            self.llf = -363.306763257476
            self.resid = residsc_css[1:, 0]
            self.fittedvalues = yhatc_css[1:, 0]
            self.pvalues = [3.51e-08, 4.7e-31, 8.35e-11]
            self.tvalues = [17.544326, 11.951494, 6.009649]
            self.sigma2 = 1.040940645447 ** 2