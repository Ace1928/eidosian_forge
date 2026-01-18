import numpy as np
class SCAD(Penalty):
    """
    The SCAD penalty of Fan and Li.

    The SCAD penalty is linear around zero as a L1 penalty up to threshold tau.
    The SCAD penalty is constant for values larger than c*tau.
    The middle segment is quadratic and connect the two segments with a continuous
    derivative.
    The penalty is symmetric around zero.

    Parameterization follows Boo, Johnson, Li and Tan 2011.
    Fan and Li use lambda instead of tau, and a instead of c. Fan and Li
    recommend setting c=3.7.

    f(x) = { tau |x|                                        if 0 <= |x| < tau
           { -(|x|^2 - 2 c tau |x| + tau^2) / (2 (c - 1))   if tau <= |x| < c tau
           { (c + 1) tau^2 / 2                              if c tau <= |x|

    Parameters
    ----------
    tau : float
        slope and threshold for linear segment
    c : float
        factor for second threshold which is c * tau
    weights : None or array
        weights for penalty of each parameter. If an entry is zero, then the
        corresponding parameter will not be penalized.

    References
    ----------
    Buu, Anne, Norman J. Johnson, Runze Li, and Xianming Tan. "New variable
    selection methods for zero‐inflated count data with applications to the
    substance abuse field."
    Statistics in medicine 30, no. 18 (2011): 2326-2340.

    Fan, Jianqing, and Runze Li. "Variable selection via nonconcave penalized
    likelihood and its oracle properties."
    Journal of the American statistical Association 96, no. 456 (2001):
    1348-1360.
    """

    def __init__(self, tau, c=3.7, weights=1.0):
        super().__init__(weights)
        self.tau = tau
        self.c = c

    def func(self, params):
        tau = self.tau
        p_abs = np.atleast_1d(np.abs(params))
        res = np.empty(p_abs.shape, p_abs.dtype)
        res.fill(np.nan)
        mask1 = p_abs < tau
        mask3 = p_abs >= self.c * tau
        res[mask1] = tau * p_abs[mask1]
        mask2 = ~mask1 & ~mask3
        p_abs2 = p_abs[mask2]
        tmp = p_abs2 ** 2 - 2 * self.c * tau * p_abs2 + tau ** 2
        res[mask2] = -tmp / (2 * (self.c - 1))
        res[mask3] = (self.c + 1) * tau ** 2 / 2.0
        return (self.weights * res).sum(0)

    def deriv(self, params):
        tau = self.tau
        p = np.atleast_1d(params)
        p_abs = np.abs(p)
        p_sign = np.sign(p)
        res = np.empty(p_abs.shape)
        res.fill(np.nan)
        mask1 = p_abs < tau
        mask3 = p_abs >= self.c * tau
        mask2 = ~mask1 & ~mask3
        res[mask1] = p_sign[mask1] * tau
        tmp = p_sign[mask2] * (p_abs[mask2] - self.c * tau)
        res[mask2] = -tmp / (self.c - 1)
        res[mask3] = 0
        return self.weights * res

    def deriv2(self, params):
        """Second derivative of function

        This returns scalar or vector in same shape as params, not a square
        Hessian. If the return is 1 dimensional, then it is the diagonal of
        the Hessian.
        """
        tau = self.tau
        p = np.atleast_1d(params)
        p_abs = np.abs(p)
        res = np.zeros(p_abs.shape)
        mask1 = p_abs < tau
        mask3 = p_abs >= self.c * tau
        mask2 = ~mask1 & ~mask3
        res[mask2] = -1 / (self.c - 1)
        return self.weights * res