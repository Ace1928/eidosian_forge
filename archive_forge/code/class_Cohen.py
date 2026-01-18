class Cohen(InverseLaplaceTransform):

    def calc_laplace_parameter(self, t, **kwargs):
        """The Cohen algorithm accelerates the convergence of the nearly
        alternating series resulting from the application of the trapezoidal
        rule to the Bromwich contour inversion integral.

        .. math ::

            p_k = \\frac{\\gamma}{2 t} + \\frac{\\pi i k}{t} \\qquad 0 \\le k < M

        where

        .. math ::

            \\gamma = \\frac{2}{3} (d + \\log(10) + \\log(2 t)),

        `d = \\mathrm{dps\\_goal}`, which is chosen based on the desired
        accuracy using the method developed in [1] to improve numerical
        stability. The Cohen algorithm shows robustness similar to the de Hoog
        et al. algorithm, but it is faster than the fixed Talbot algorithm.

        **Optional arguments**

        *degree*
            integer order of the approximation (M = number of terms)
        *alpha*
            abscissa for `p_0` (controls the discretization error)

        The working precision will be increased according to a rule of
        thumb. If 'degree' is not specified, the working precision and
        degree are chosen to hopefully achieve the dps of the calling
        context. If 'degree' is specified, the working precision is
        chosen to achieve maximum resulting precision for the
        specified degree.

        **References**

        1. P. Glasserman, J. Ruiz-Mata (2006). Computing the credit loss
        distribution in the Gaussian copula model: a comparison of methods.
        *Journal of Credit Risk* 2(4):33-66, 10.21314/JCR.2006.057

        """
        self.t = self.ctx.convert(t)
        if 'degree' in kwargs:
            self.degree = kwargs['degree']
            self.dps_goal = int(1.5 * self.degree)
        else:
            self.dps_goal = int(self.ctx.dps * 1.74)
            self.degree = max(22, int(1.31 * self.dps_goal))
        M = self.degree + 1
        self.dps_orig = self.ctx.dps
        self.ctx.dps = self.dps_goal
        ttwo = 2 * self.t
        tmp = self.ctx.dps * self.ctx.log(10) + self.ctx.log(ttwo)
        tmp = self.ctx.fraction(2, 3) * tmp
        self.alpha = self.ctx.convert(kwargs.get('alpha', tmp))
        a_t = self.alpha / ttwo
        p_t = self.ctx.pi * 1j / self.t
        self.p = self.ctx.matrix(M, 1)
        self.p[0] = a_t
        for i in range(1, M):
            self.p[i] = a_t + i * p_t

    def calc_time_domain_solution(self, fp, t, manual_prec=False):
        """Calculate time-domain solution for Cohen algorithm.

        The accelerated nearly alternating series is:

        .. math ::

            f(t, M) = \\frac{e^{\\gamma / 2}}{t} \\left[\\frac{1}{2}
            \\Re\\left(\\bar{f}\\left(\\frac{\\gamma}{2t}\\right) \\right) -
            \\sum_{k=0}^{M-1}\\frac{c_{M,k}}{d_M}\\Re\\left(\\bar{f}
            \\left(\\frac{\\gamma + 2(k+1) \\pi i}{2t}\\right)\\right)\\right],

        where coefficients `\\frac{c_{M, k}}{d_M}` are described in [1].

        1. H. Cohen, F. Rodriguez Villegas, D. Zagier (2000). Convergence
        acceleration of alternating series. *Experiment. Math* 9(1):3-12

        """
        self.t = self.ctx.convert(t)
        n = self.degree
        M = n + 1
        A = self.ctx.matrix(M, 1)
        for i in range(M):
            A[i] = fp[i].real
        d = (3 + self.ctx.sqrt(8)) ** n
        d = (d + 1 / d) / 2
        b = -self.ctx.one
        c = -d
        s = 0
        for k in range(n):
            c = b - c
            s = s + c * A[k + 1]
            b = 2 * (k + n) * (k - n) * b / ((2 * k + 1) * (k + self.ctx.one))
        result = self.ctx.exp(self.alpha / 2) / self.t * (A[0] / 2 - s / d)
        if not manual_prec:
            self.ctx.dps = self.dps_orig
        return result