class deHoog(InverseLaplaceTransform):

    def calc_laplace_parameter(self, t, **kwargs):
        """the de Hoog, Knight & Stokes algorithm is an
        accelerated form of the Fourier series numerical
        inverse Laplace transform algorithms.

        .. math ::

            p_k = \\gamma + \\frac{jk}{T} \\qquad 0 \\le k < 2M+1

        where

        .. math ::

            \\gamma = \\alpha - \\frac{\\log \\mathrm{tol}}{2T},

        `j=\\sqrt{-1}`, `T = 2t_\\mathrm{max}` is a scaled time,
        `\\alpha=10^{-\\mathrm{dps\\_goal}}` is the real part of the
        rightmost pole or singularity, which is chosen based on the
        desired accuracy (assuming the rightmost singularity is 0),
        and `\\mathrm{tol}=10\\alpha` is the desired tolerance, which is
        chosen in relation to `\\alpha`.`

        When increasing the degree, the abscissa increase towards
        `j\\infty`, but more slowly than the fixed Talbot
        algorithm. The de Hoog et al. algorithm typically does better
        with oscillatory functions of time, and less well-behaved
        functions. The method tends to be slower than the Talbot and
        Stehfest algorithsm, especially so at very high precision
        (e.g., `>500` digits precision).

        """
        self.t = self.ctx.convert(t)
        self.tmax = kwargs.get('tmax', self.t)
        if 'degree' in kwargs:
            self.degree = kwargs['degree']
            self.dps_goal = int(1.38 * self.degree)
        else:
            self.dps_goal = int(self.ctx.dps * 1.36)
            self.degree = max(10, self.dps_goal)
        M = self.degree
        tmp = self.ctx.power(10.0, -self.dps_goal)
        self.alpha = self.ctx.convert(kwargs.get('alpha', tmp))
        self.tol = self.ctx.convert(kwargs.get('tol', self.alpha * 10.0))
        self.np = 2 * self.degree + 1
        self.dps_orig = self.ctx.dps
        self.ctx.dps = self.dps_goal
        self.scale = kwargs.get('scale', 2)
        self.T = self.ctx.convert(kwargs.get('T', self.scale * self.tmax))
        self.p = self.ctx.matrix(2 * M + 1, 1)
        self.gamma = self.alpha - self.ctx.log(self.tol) / (self.scale * self.T)
        self.p = self.gamma + self.ctx.pi * self.ctx.matrix(self.ctx.arange(self.np)) / self.T * 1j

    def calc_time_domain_solution(self, fp, t, manual_prec=False):
        """Calculate time-domain solution for
        de Hoog, Knight & Stokes algorithm.

        The un-accelerated Fourier series approach is:

        .. math ::

            f(t,2M+1) = \\frac{e^{\\gamma t}}{T} \\sum_{k=0}^{2M}{}^{'}
            \\Re\\left[\\bar{f}\\left( p_k \\right)
            e^{i\\pi t/T} \\right],

        where the prime on the summation indicates the first term is halved.

        This simplistic approach requires so many function evaluations
        that it is not practical. Non-linear acceleration is
        accomplished via Pade-approximation and an analytic expression
        for the remainder of the continued fraction. See the original
        paper (reference 2 below) a detailed description of the
        numerical approach.

        **References**

        1. Davies, B. (2005). *Integral Transforms and their
           Applications*, Third Edition. Springer.
        2. de Hoog, F., J. Knight, A. Stokes (1982). An improved
           method for numerical inversion of Laplace transforms. *SIAM
           Journal of Scientific and Statistical Computing* 3:357-366,
           http://dx.doi.org/10.1137/0903022

        """
        M = self.degree
        np = self.np
        T = self.T
        self.t = self.ctx.convert(t)
        e = self.ctx.zeros(np, M + 1)
        q = self.ctx.matrix(2 * M, M)
        d = self.ctx.matrix(np, 1)
        A = self.ctx.zeros(np + 1, 1)
        B = self.ctx.ones(np + 1, 1)
        e[:, 0] = 0.0 + 0j
        q[0, 0] = fp[1] / (fp[0] / 2)
        for i in range(1, 2 * M):
            q[i, 0] = fp[i + 1] / fp[i]
        for r in range(1, M + 1):
            mr = 2 * (M - r) + 1
            e[0:mr, r] = q[1:mr + 1, r - 1] - q[0:mr, r - 1] + e[1:mr + 1, r - 1]
            if not r == M:
                rq = r + 1
                mr = 2 * (M - rq) + 1 + 2
                for i in range(mr):
                    q[i, rq - 1] = q[i + 1, rq - 2] * e[i + 1, rq - 1] / e[i, rq - 1]
        d[0] = fp[0] / 2
        for r in range(1, M + 1):
            d[2 * r - 1] = -q[0, r - 1]
            d[2 * r] = -e[0, r]
        A[0] = 0.0 + 0j
        A[1] = d[0]
        B[0:2] = 1.0 + 0j
        z = self.ctx.expjpi(self.t / T)
        for i in range(1, 2 * M):
            A[i + 1] = A[i] + d[i] * A[i - 1] * z
            B[i + 1] = B[i] + d[i] * B[i - 1] * z
        brem = (1 + (d[2 * M - 1] - d[2 * M]) * z) / 2
        rem = brem * self.ctx.powm1(1 + d[2 * M] * z / brem, self.ctx.fraction(1, 2))
        A[np] = A[2 * M] + rem * A[2 * M - 1]
        B[np] = B[2 * M] + rem * B[2 * M - 1]
        result = self.ctx.exp(self.gamma * self.t) / T * (A[np] / B[np]).real
        if not manual_prec:
            self.ctx.dps = self.dps_orig
        return result