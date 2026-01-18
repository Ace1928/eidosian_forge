import numpy as np
from scipy import stats
Summarize the Results of the nonlinear transformation.

        This provides a parameter table equivalent to `t_test` and reuses
        `ContrastResults`.

        Parameters
        -----------
        xname : list of strings, optional
            Default is `c_##` for ## in p the number of regressors
        alpha : float
            Significance level for the confidence intervals. Default is
            alpha = 0.05 which implies a confidence level of 95%.
        title : string, optional
            Title for the params table. If not None, then this replaces the
            default title
        use_t : boolean
            If use_t is False (default), then the normal distribution is used
            for the confidence interval, otherwise the t distribution with
            `df` degrees of freedom is used.
        df : int or float
            degrees of freedom for t distribution. Only used and required if
            use_t is True.

        Returns
        -------
        smry : string or Summary instance
            This contains a parameter results table in the case of t or z test
            in the same form as the parameter results table in the model
            results summary.
            For F or Wald test, the return is a string.
        