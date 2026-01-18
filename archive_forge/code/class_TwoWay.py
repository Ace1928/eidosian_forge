from numpy.testing import assert_equal
import numpy as np
class TwoWay:
    """a wrapper class for two way anova type of analysis with OLS


    currently mainly to bring things together

    Notes
    -----
    unclear: adding multiple test might assume block design or orthogonality

    This estimates the full dummy version with OLS.
    The drop first dummy representation can be recovered through the
    transform method.

    TODO: add more methods, tests, pairwise, multiple, marginal effects
    try out what can be added for userfriendly access.

    missing: ANOVA table

    """

    def __init__(self, endog, factor1, factor2, varnames=None):
        self.nobs = factor1.shape[0]
        if varnames is None:
            vname1 = 'a'
            vname2 = 'b'
        else:
            vname1, vname1 = varnames
        self.d1, self.d1_labels = d1, d1_labels = dummy_1d(factor1, vname1)
        self.d2, self.d2_labels = d2, d2_labels = dummy_1d(factor2, vname2)
        self.nlevel1 = nlevel1 = d1.shape[1]
        self.nlevel2 = nlevel2 = d2.shape[1]
        res = contrast_product(d1_labels, d2_labels)
        prodlab, C1, C1lab, C2, C2lab, _ = res
        self.prod_label, self.C1, self.C1_label, self.C2, self.C2_label, _ = res
        dp_full = dummy_product(d1, d2, method='full')
        dp_dropf = dummy_product(d1, d2, method='drop-first')
        self.transform = DummyTransform(dp_full, dp_dropf)
        self.nvars = dp_full.shape[1]
        self.exog = dp_full
        self.resols = sm.OLS(endog, dp_full).fit()
        self.params = self.resols.params
        self.params_dropf = self.transform.inv_dot_left(self.params)
        self.start_interaction = 1 + (nlevel1 - 1) + (nlevel2 - 1)
        self.n_interaction = self.nvars - self.start_interaction

    def r_nointer(self):
        """contrast/restriction matrix for no interaction
        """
        nia = self.n_interaction
        R_nointer = np.hstack((np.zeros((nia, self.nvars - nia)), np.eye(nia)))
        R_nointer_transf = self.transform.inv_dot_right(R_nointer)
        self.R_nointer_transf = R_nointer_transf
        return R_nointer_transf

    def ttest_interaction(self):
        """ttests for no-interaction terms are zero
        """
        nia = self.n_interaction
        R_nointer = np.hstack((np.zeros((nia, self.nvars - nia)), np.eye(nia)))
        R_nointer_transf = self.transform.inv_dot_right(R_nointer)
        self.R_nointer_transf = R_nointer_transf
        t_res = self.resols.t_test(R_nointer_transf)
        return t_res

    def ftest_interaction(self):
        """ttests for no-interaction terms are zero
        """
        R_nointer_transf = self.r_nointer()
        return self.resols.f_test(R_nointer_transf)

    def ttest_conditional_effect(self, factorind):
        if factorind == 1:
            return (self.resols.t_test(self.C1), self.C1_label)
        else:
            return (self.resols.t_test(self.C2), self.C2_label)

    def summary_coeff(self):
        from statsmodels.iolib import SimpleTable
        params_arr = self.params.reshape(self.nlevel1, self.nlevel2)
        stubs = self.d1_labels
        headers = self.d2_labels
        title = 'Estimated Coefficients by factors'
        table_fmt = dict(data_fmts=['%#10.4g'] * self.nlevel2)
        return SimpleTable(params_arr, headers, stubs, title=title, txt_fmt=table_fmt)