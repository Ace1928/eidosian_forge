import numpy as np
class ForwardEffects(RegressionEffects):
    """
    Forward selection effect sizes for FDR control.

    Parameters
    ----------
    parent : RegressionFDR
        The RegressionFDR instance to which this effect size is
        applied.
    pursuit : bool
        If True, 'basis pursuit' is used, which amounts to performing
        a full regression at each selection step to adjust the working
        residual vector.  If False (the default), the residual is
        adjusted by regressing out each selected variable marginally.
        Setting pursuit=True will be considerably slower, but may give
        better results when exog is not orthogonal.

    Notes
    -----
    This class implements the forward selection approach to
    constructing test statistics for a knockoff analysis, as
    described under (5) in section 2.2 of the Barber and Candes
    paper.
    """

    def __init__(self, pursuit):
        self.pursuit = pursuit

    def stats(self, parent):
        nvar = parent.exog.shape[1]
        rv = parent.endog.copy()
        vl = [(i, parent.exog[:, i]) for i in range(nvar)]
        z = np.empty(nvar)
        past = []
        for i in range(nvar):
            dp = np.r_[[np.abs(np.dot(rv, x[1])) for x in vl]]
            j = np.argmax(dp)
            z[vl[j][0]] = nvar - i - 1
            x = vl[j][1]
            del vl[j]
            if self.pursuit:
                for v in past:
                    x -= np.dot(x, v) * v
                past.append(x)
            rv -= np.dot(rv, x) * x
        z1 = z[0:nvar // 2]
        z2 = z[nvar // 2:]
        st = np.where(z1 > z2, z1, z2) * np.sign(z1 - z2)
        return st