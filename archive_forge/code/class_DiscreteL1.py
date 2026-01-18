import os
import numpy as np
class DiscreteL1:

    def __init__(self):
        """
        Special results for L1 models
        Uses the Spector data and a script to generate the baseline results
        """
        pass

    def logit():
        """
        Results generated with:
            data = sm.datasets.spector.load()
            data.exog = sm.add_constant(data.exog, prepend=True)
            alpha = 3 * np.array([0, 1, 1, 1])
            res2 = sm.Logit(data.endog, data.exog).fit_regularized(
                method="l1", alpha=alpha, disp=0, trim_mode='size',
                size_trim_tol=1e-5, acc=1e-10, maxiter=1000)
        """
        obj = Namespace()
        nan = np.nan
        obj.params = [-4.10271595, 0.0, 0.15493781, 0.0]
        obj.conf_int = [[-9.15205122, 0.94661932], [nan, nan], [-0.06539482, 0.37527044], [nan, nan]]
        obj.bse = [2.5762388, nan, 0.11241668, nan]
        obj.nnz_params = 2
        obj.aic = 42.09143936858367
        obj.bic = 45.02291117418312
        obj.cov_params = [[6.63700638, nan, -0.28636261, nan], [nan, nan, nan, nan], [-0.28636261, nan, 0.01263751, nan], [nan, nan, nan, nan]]
        return obj
    logit = logit()

    def sweep():
        """
        Results generated with
            params = np.zeros((3, 4))
            alphas = np.array(
                    [[0.1, 0.1, 0.1, 0.1],
                        [0.4, 0.4, 0.5, 0.5], [0.5, 0.5, 1, 1]])
            model = sm.Logit(data.endog, data.exog)
            for i in range(3):
                alpha = alphas[i, :]
                res2 = model.fit_regularized(method="l1", alpha=alpha,
                                             disp=0, acc=1e-10,
                                             maxiter=1000, trim_mode='off')
                params[i, :] = res2.params
            print(params)
        """
        obj = Namespace()
        obj.params = [[-10.37593611, 2.27080968, 0.06670638, 2.05723691], [-5.32670811, 1.18216019, 0.01402395, 1.45178712], [-3.92630318, 0.90126958, -0.0, 1.09498178]]
        return obj
    sweep = sweep()

    def probit():
        """
        Results generated with
            data = sm.datasets.spector.load()
            data.exog = sm.add_constant(data.exog, prepend=True)
            alpha = np.array([0.1, 0.2, 0.3, 10])
            res2 = sm.Probit(data.endog, data.exog).fit_regularized(
                method="l1", alpha=alpha, disp=0, trim_mode='auto',
                auto_trim_tol=0.02, acc=1e-10, maxiter=1000)
        """
        obj = Namespace()
        nan = np.nan
        obj.params = [-5.40476992, 1.25018458, 0.04744558, 0.0]
        obj.conf_int = [[-9.44077951, -1.36876033], [0.03716721, 2.46320194], [-0.09727571, 0.19216687], [np.nan, np.nan]]
        obj.bse = [2.05922641, 0.61889778, 0.07383875, np.nan]
        obj.nnz_params = 3
        obj.aic = 38.39977387754293
        obj.bic = 42.796981585942106
        obj.cov_params = [[4.24041339, -0.83432592, -0.06827915, nan], [-0.83432592, 0.38303447, -0.01700249, nan], [-0.06827915, -0.01700249, 0.00545216, nan], [nan, nan, nan, nan]]
        return obj
    probit = probit()

    def mnlogit():
        """
        Results generated with
            anes_data = sm.datasets.anes96.load()
            anes_exog = anes_data.exog
            anes_exog = sm.add_constant(anes_exog, prepend=False)
            mlogit_mod = sm.MNLogit(anes_data.endog, anes_exog)

            alpha = 10 * np.ones((mlogit_mod.J - 1, mlogit_mod.K))
            alpha[-1, :] = 0
            mlogit_l1_res = mlogit_mod.fit_regularized(
            method='l1', alpha=alpha, trim_mode='auto', auto_trim_tol=0.02,
            acc=1e-10)
        """
        obj = Namespace()
        obj.params = [[0.00100163, -0.05864195, -0.06147822, -0.04769671, -0.05222987, -0.09522432], [0.0, 0.03186139, 0.12048999, 0.83211915, 0.92330292, 1.5680646], [-0.0218185, -0.01988066, -0.00808564, -0.00487463, -0.01400173, -0.00562079], [0.0, 0.03306875, 0.0, 0.02362861, 0.05486435, 0.14656966], [0.0, 0.04448213, 0.03252651, 0.07661761, 0.07265266, 0.0967758], [0.90993803, -0.50081247, -2.08285102, -5.26132955, -4.86783179, -9.31537963]]
        obj.conf_int = [[[-0.0646223, 0.06662556], [np.nan, np.nan], [-0.03405931, -0.00957768], [np.nan, np.nan], [np.nan, np.nan], [0.26697895, 1.55289711]], [[-0.1337913, 0.01650741], [-0.14477255, 0.20849532], [-0.03500303, -0.00475829], [-0.11406121, 0.18019871], [0.00479741, 0.08416684], [-1.84626136, 0.84463642]], [[-0.17237962, 0.04942317], [-0.15146029, 0.39244026], [-0.02947379, 0.01330252], [np.nan, np.nan], [-0.02501483, 0.09006785], [-3.90379391, -0.26190812]], [[-0.12938296, 0.03398954], [0.62612955, 1.03810876], [-0.02046322, 0.01071395], [-0.13738534, 0.18464256], [0.03017236, 0.12306286], [-6.91227465, -3.61038444]], [[-0.12469773, 0.02023799], [0.742564, 1.10404183], [-0.02791975, -8.371e-05], [-0.08491561, 0.19464431], [0.0332926, 0.11201273], [-6.29331126, -3.44235233]], [[-0.17165567, -0.01879296], [1.33994079, 1.79618841], [-0.02027503, 0.00903345], [-0.00267819, 0.29581751], [0.05343135, 0.14012026], [-11.10419107, -7.52656819]]]
        obj.bse = [[0.03348221, 0.03834221, 0.05658338, 0.04167742, 0.03697408, 0.03899631], [np.nan, 0.09012101, 0.13875269, 0.10509867, 0.09221543, 0.11639184], [0.00624543, 0.00771564, 0.01091253, 0.00795351, 0.00710116, 0.00747679], [np.nan, 0.07506769, np.nan, 0.08215148, 0.07131762, 0.07614826], [np.nan, 0.02024768, 0.02935837, 0.02369699, 0.02008204, 0.02211492], [0.32804638, 0.68646613, 0.92906957, 0.84233441, 0.72729881, 0.91267567]]
        obj.nnz_params = 32
        obj.aic = 3019.4391360294126
        obj.bic = 3174.6431733460686
        return obj
    mnlogit = mnlogit()