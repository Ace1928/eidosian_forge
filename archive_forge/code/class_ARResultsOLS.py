import os
import numpy as np
class ARResultsOLS:
    """
    Results of fitting an AR(9) model to the sunspot data.

    Results were taken from Stata using the var command.
   """

    def __init__(self, constant=True):
        self.avobs = 300.0
        if constant:
            self.params = [6.7430535917332, 1.1649421971129, -0.40535742259304, -0.16653934246587, 0.14980629416032, -0.09462417064796, 0.00491001240749, 0.0504665930841, -0.08635349190816, 0.25349103194757]
            self.bse_stata = [2.413485601, 0.0560359041, 0.0874490762, 0.0900894414, 0.0899348339, 0.0900100797, 0.0898385666, 0.0896997939, 0.0869773089, 0.0559505756]
            self.bse_gretl = [2.45474, 0.0569939, 0.088944, 0.0916295, 0.0914723, 0.0915488, 0.0913744, 0.0912332, 0.0884642, 0.0569071]
            self.rmse = 15.1279294937327
            self.fpe = 236.4827257929261
            self.llf = -1235.559128419549
            filename = os.path.join(cur_dir, 'AROLSConstantPredict.csv')
            predictresults = np.loadtxt(filename)
            fv = predictresults[:300, 0]
            pv = predictresults[300:, 1]
            del predictresults
            self.FVOLSnneg1start0 = fv
            self.FVOLSnneg1start9 = fv
            self.FVOLSnneg1start100 = fv[100 - 9:]
            self.FVOLSn200start0 = fv[:192]
            self.FVOLSn200start200 = np.hstack((fv[200 - 9:], pv[:101 - 9]))
            self.FVOLSn200startneg109 = self.FVOLSn200start200
            self.FVOLSn100start325 = np.hstack((fv[-1], pv))
            self.FVOLSn301start9 = np.hstack((fv, pv[:2]))
            self.FVOLSdefault = fv
            self.FVOLSn4start312 = np.hstack((fv[-1], pv[:8]))
            self.FVOLSn15start312 = np.hstack((fv[-1], pv[:19]))
        elif not constant:
            self.params = [1.19582389902985, -0.40591818219637, -0.15813796884843, 0.16620079925202, -0.08570200254617, 0.01876298948686, 0.06130211910707, -0.08461507700047, 0.27995084653313]
            self.bse_stata = [0.055645055, 0.088579237, 0.0912031179, 0.0909032462, 0.0911161784, 0.0908611473, 0.0907743174, 0.0880993504, 0.0558560278]
            self.bse_gretl = [0.056499, 0.0899386, 0.0926027, 0.0922983, 0.0925145, 0.0922555, 0.0921674, 0.0894513, 0.0567132]
            self.rmse = 15.29712618677774
            self.sigma = 226.9820074869752
            self.llf = -1239.41217278661
            self.fpe = 241.0221316614273
            filename = os.path.join(cur_dir, 'AROLSNoConstantPredict.csv')
            predictresults = np.loadtxt(filename)
            fv = predictresults[:300, 0]
            pv = predictresults[300:, 1]
            del predictresults
            self.FVOLSnneg1start0 = fv
            self.FVOLSnneg1start9 = fv
            self.FVOLSnneg1start100 = fv[100 - 9:]
            self.FVOLSn200start0 = fv[:192]
            self.FVOLSn200start200 = np.hstack((fv[200 - 9:], pv[:101 - 9]))
            self.FVOLSn200startneg109 = self.FVOLSn200start200
            self.FVOLSn100start325 = np.hstack((fv[-1], pv))
            self.FVOLSn301start9 = np.hstack((fv, pv[:2]))
            self.FVOLSdefault = fv
            self.FVOLSn4start312 = np.hstack((fv[-1], pv[:8]))
            self.FVOLSn15start312 = np.hstack((fv[-1], pv[:19]))