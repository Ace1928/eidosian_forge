import math
from ..libmp.backend import xrange
class QuadratureRule(object):
    """
    Quadrature rules are implemented using this class, in order to
    simplify the code and provide a common infrastructure
    for tasks such as error estimation and node caching.

    You can implement a custom quadrature rule by subclassing
    :class:`QuadratureRule` and implementing the appropriate
    methods. The subclass can then be used by :func:`~mpmath.quad` by
    passing it as the *method* argument.

    :class:`QuadratureRule` instances are supposed to be singletons.
    :class:`QuadratureRule` therefore implements instance caching
    in :func:`~mpmath.__new__`.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.standard_cache = {}
        self.transformed_cache = {}
        self.interval_count = {}

    def clear(self):
        """
        Delete cached node data.
        """
        self.standard_cache = {}
        self.transformed_cache = {}
        self.interval_count = {}

    def calc_nodes(self, degree, prec, verbose=False):
        """
        Compute nodes for the standard interval `[-1, 1]`. Subclasses
        should probably implement only this method, and use
        :func:`~mpmath.get_nodes` method to retrieve the nodes.
        """
        raise NotImplementedError

    def get_nodes(self, a, b, degree, prec, verbose=False):
        """
        Return nodes for given interval, degree and precision. The
        nodes are retrieved from a cache if already computed;
        otherwise they are computed by calling :func:`~mpmath.calc_nodes`
        and are then cached.

        Subclasses should probably not implement this method,
        but just implement :func:`~mpmath.calc_nodes` for the actual
        node computation.
        """
        key = (a, b, degree, prec)
        if key in self.transformed_cache:
            return self.transformed_cache[key]
        orig = self.ctx.prec
        try:
            self.ctx.prec = prec + 20
            if (degree, prec) in self.standard_cache:
                nodes = self.standard_cache[degree, prec]
            else:
                nodes = self.calc_nodes(degree, prec, verbose)
                self.standard_cache[degree, prec] = nodes
            nodes = self.transform_nodes(nodes, a, b, verbose)
            if key in self.interval_count:
                self.transformed_cache[key] = nodes
            else:
                self.interval_count[key] = True
        finally:
            self.ctx.prec = orig
        return nodes

    def transform_nodes(self, nodes, a, b, verbose=False):
        """
        Rescale standardized nodes (for `[-1, 1]`) to a general
        interval `[a, b]`. For a finite interval, a simple linear
        change of variables is used. Otherwise, the following
        transformations are used:

        .. math ::

            \\lbrack a, \\infty \\rbrack : t = \\frac{1}{x} + (a-1)

            \\lbrack -\\infty, b \\rbrack : t = (b+1) - \\frac{1}{x}

            \\lbrack -\\infty, \\infty \\rbrack : t = \\frac{x}{\\sqrt{1-x^2}}

        """
        ctx = self.ctx
        a = ctx.convert(a)
        b = ctx.convert(b)
        one = ctx.one
        if (a, b) == (-one, one):
            return nodes
        half = ctx.mpf(0.5)
        new_nodes = []
        if ctx.isinf(a) or ctx.isinf(b):
            if (a, b) == (ctx.ninf, ctx.inf):
                p05 = -half
                for x, w in nodes:
                    x2 = x * x
                    px1 = one - x2
                    spx1 = px1 ** p05
                    x = x * spx1
                    w *= spx1 / px1
                    new_nodes.append((x, w))
            elif a == ctx.ninf:
                b1 = b + 1
                for x, w in nodes:
                    u = 2 / (x + one)
                    x = b1 - u
                    w *= half * u ** 2
                    new_nodes.append((x, w))
            elif b == ctx.inf:
                a1 = a - 1
                for x, w in nodes:
                    u = 2 / (x + one)
                    x = a1 + u
                    w *= half * u ** 2
                    new_nodes.append((x, w))
            elif a == ctx.inf or b == ctx.ninf:
                return [(x, -w) for x, w in self.transform_nodes(nodes, b, a, verbose)]
            else:
                raise NotImplementedError
        else:
            C = (b - a) / 2
            D = (b + a) / 2
            for x, w in nodes:
                new_nodes.append((D + C * x, C * w))
        return new_nodes

    def guess_degree(self, prec):
        """
        Given a desired precision `p` in bits, estimate the degree `m`
        of the quadrature required to accomplish full accuracy for
        typical integrals. By default, :func:`~mpmath.quad` will perform up
        to `m` iterations. The value of `m` should be a slight
        overestimate, so that "slightly bad" integrals can be dealt
        with automatically using a few extra iterations. On the
        other hand, it should not be too big, so :func:`~mpmath.quad` can
        quit within a reasonable amount of time when it is given
        an "unsolvable" integral.

        The default formula used by :func:`~mpmath.guess_degree` is tuned
        for both :class:`TanhSinh` and :class:`GaussLegendre`.
        The output is roughly as follows:

            +---------+---------+
            | `p`     | `m`     |
            +=========+=========+
            | 50      | 6       |
            +---------+---------+
            | 100     | 7       |
            +---------+---------+
            | 500     | 10      |
            +---------+---------+
            | 3000    | 12      |
            +---------+---------+

        This formula is based purely on a limited amount of
        experimentation and will sometimes be wrong.
        """
        g = int(4 + max(0, self.ctx.log(prec / 30.0, 2)))
        g += 2
        return g

    def estimate_error(self, results, prec, epsilon):
        """
        Given results from integrations `[I_1, I_2, \\ldots, I_k]` done
        with a quadrature of rule of degree `1, 2, \\ldots, k`, estimate
        the error of `I_k`.

        For `k = 2`, we estimate  `|I_{\\infty}-I_2|` as `|I_2-I_1|`.

        For `k > 2`, we extrapolate `|I_{\\infty}-I_k| \\approx |I_{k+1}-I_k|`
        from `|I_k-I_{k-1}|` and `|I_k-I_{k-2}|` under the assumption
        that each degree increment roughly doubles the accuracy of
        the quadrature rule (this is true for both :class:`TanhSinh`
        and :class:`GaussLegendre`). The extrapolation formula is given
        by Borwein, Bailey & Girgensohn. Although not very conservative,
        this method seems to be very robust in practice.
        """
        if len(results) == 2:
            return abs(results[0] - results[1])
        try:
            if results[-1] == results[-2] == results[-3]:
                return self.ctx.zero
            D1 = self.ctx.log(abs(results[-1] - results[-2]), 10)
            D2 = self.ctx.log(abs(results[-1] - results[-3]), 10)
        except ValueError:
            return epsilon
        D3 = -prec
        D4 = min(0, max(D1 ** 2 / D2, 2 * D1, D3))
        return self.ctx.mpf(10) ** int(D4)

    def summation(self, f, points, prec, epsilon, max_degree, verbose=False):
        """
        Main integration function. Computes the 1D integral over
        the interval specified by *points*. For each subinterval,
        performs quadrature of degree from 1 up to *max_degree*
        until :func:`~mpmath.estimate_error` signals convergence.

        :func:`~mpmath.summation` transforms each subintegration to
        the standard interval and then calls :func:`~mpmath.sum_next`.
        """
        ctx = self.ctx
        I = total_err = ctx.zero
        for i in xrange(len(points) - 1):
            a, b = (points[i], points[i + 1])
            if a == b:
                continue
            if (a, b) == (ctx.ninf, ctx.inf):
                _f = f
                f = lambda x: _f(-x) + _f(x)
                a, b = (ctx.zero, ctx.inf)
            results = []
            err = ctx.zero
            for degree in xrange(1, max_degree + 1):
                nodes = self.get_nodes(a, b, degree, prec, verbose)
                if verbose:
                    print('Integrating from %s to %s (degree %s of %s)' % (ctx.nstr(a), ctx.nstr(b), degree, max_degree))
                result = self.sum_next(f, nodes, degree, prec, results, verbose)
                results.append(result)
                if degree > 1:
                    err = self.estimate_error(results, prec, epsilon)
                    if verbose:
                        print('Estimated error:', ctx.nstr(err), ' epsilon:', ctx.nstr(epsilon), ' result: ', ctx.nstr(result))
                    if err <= epsilon:
                        break
            I += results[-1]
            total_err += err
        if total_err > epsilon:
            if verbose:
                print('Failed to reach full accuracy. Estimated error:', ctx.nstr(total_err))
        return (I, total_err)

    def sum_next(self, f, nodes, degree, prec, previous, verbose=False):
        """
        Evaluates the step sum `\\sum w_k f(x_k)` where the *nodes* list
        contains the `(w_k, x_k)` pairs.

        :func:`~mpmath.summation` will supply the list *results* of
        values computed by :func:`~mpmath.sum_next` at previous degrees, in
        case the quadrature rule is able to reuse them.
        """
        return self.ctx.fdot(((w, f(x)) for x, w in nodes))