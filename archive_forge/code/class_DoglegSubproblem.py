import numpy as np
import scipy.linalg
from ._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)
class DoglegSubproblem(BaseQuadraticSubproblem):
    """Quadratic subproblem solved by the dogleg method"""

    def cauchy_point(self):
        """
        The Cauchy point is minimal along the direction of steepest descent.
        """
        if self._cauchy_point is None:
            g = self.jac
            Bg = self.hessp(g)
            self._cauchy_point = -(np.dot(g, g) / np.dot(g, Bg)) * g
        return self._cauchy_point

    def newton_point(self):
        """
        The Newton point is a global minimum of the approximate function.
        """
        if self._newton_point is None:
            g = self.jac
            B = self.hess
            cho_info = scipy.linalg.cho_factor(B)
            self._newton_point = -scipy.linalg.cho_solve(cho_info, g)
        return self._newton_point

    def solve(self, trust_radius):
        """
        Minimize a function using the dog-leg trust-region algorithm.

        This algorithm requires function values and first and second derivatives.
        It also performs a costly Hessian decomposition for most iterations,
        and the Hessian is required to be positive definite.

        Parameters
        ----------
        trust_radius : float
            We are allowed to wander only this far away from the origin.

        Returns
        -------
        p : ndarray
            The proposed step.
        hits_boundary : bool
            True if the proposed step is on the boundary of the trust region.

        Notes
        -----
        The Hessian is required to be positive definite.

        References
        ----------
        .. [1] Jorge Nocedal and Stephen Wright,
               Numerical Optimization, second edition,
               Springer-Verlag, 2006, page 73.
        """
        p_best = self.newton_point()
        if scipy.linalg.norm(p_best) < trust_radius:
            hits_boundary = False
            return (p_best, hits_boundary)
        p_u = self.cauchy_point()
        p_u_norm = scipy.linalg.norm(p_u)
        if p_u_norm >= trust_radius:
            p_boundary = p_u * (trust_radius / p_u_norm)
            hits_boundary = True
            return (p_boundary, hits_boundary)
        _, tb = self.get_boundaries_intersections(p_u, p_best - p_u, trust_radius)
        p_boundary = p_u + tb * (p_best - p_u)
        hits_boundary = True
        return (p_boundary, hits_boundary)