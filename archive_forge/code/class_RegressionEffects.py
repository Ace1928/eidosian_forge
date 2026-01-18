import numpy as np
class RegressionEffects:
    """
    Base class for regression effects used in RegressionFDR.

    Any implementation of the class must provide a method called
    'stats' that takes a RegressionFDR object and returns effect sizes
    for the model coefficients.  Greater values for these statistics
    imply greater evidence that the effect is real.

    Knockoff effect sizes are based on fitting the regression model to
    an extended design matrix [X X'], where X' is a design matrix with
    the same shape as the actual design matrix X.  The construction of
    X' guarantees that there are no true associations between the
    columns of X' and the dependent variable of the regression.  If X
    has p columns, then the effect size of covariate j is based on the
    strength of the estimated association for coefficient j compared
    to the strength of the estimated association for coefficient p+j.
    """

    def stats(self, parent):
        raise NotImplementedError