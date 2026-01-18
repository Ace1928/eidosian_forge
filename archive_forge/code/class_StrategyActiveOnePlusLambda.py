import copy
from math import sqrt, log, exp
from itertools import cycle
import warnings
import numpy
from . import tools
class StrategyActiveOnePlusLambda(object):
    """A CMA-ES strategy that combines the :math:`(1 + \\lambda)` paradigm
    [Igel2007]_, the mixed integer modification [Hansen2011]_, active
    covariance update [Arnold2010]_ and constraint handling [Arnold2012]_.
    This version of CMA-ES requires the random vector and the mutation
    that created each individual. The vector and mutation are stored in each
    individual as :attr:`_z` and :attr:`_y` respectively. Updating with
    individuals not containing these attributes will result in an
    :class:`AttributeError`.
    Notes:
        When using this strategy (especially when using constraints) you should
        monitor the strategy :attr:`condition_number`. If it goes above a given
        threshold (say :math:`10^{12}`), you should think of restarting the
        optimization as the covariance matrix is going degenerate. See the
        constrained active CMA-ES example for a simple example of restart.
    :param parent: An iterable object that indicates where to start the
                   evolution. The parent requires a fitness attribute.
    :param sigma: The initial standard deviation of the distribution.
    :param step: The minimal step size for each dimension. Use 0 for
                 continuous dimensions.
    :param lambda_: Number of offspring to produce from the parent.
                    (optional, defaults to 1)
    :param **kwargs: One or more parameter to pass to the strategy as
                     described in the following table. (optional)
    +----------------+---------------------------+------------------------------+
    | Parameter      | Default                   | Details                      |
    +================+===========================+==============================+
    | ``d``          | ``1.0 + N / (2.0 *        | Damping for step-size.       |
    |                | lambda_)``                |                              |
    +----------------+---------------------------+------------------------------+
    | ``ptarg``      | ``1.0 / (5 + sqrt(lambda_)| Taget success rate           |
    |                | / 2.0)``                  | (from 1 + lambda algorithm). |
    +----------------+---------------------------+------------------------------+
    | ``cp``         | ``ptarg * lambda_ / (2.0 +| Step size learning rate.     |
    |                | ptarg * lambda_)``        |                              |
    +----------------+---------------------------+------------------------------+
    | ``cc``         | ``2.0 / (N + 2.0)``       | Cumulation time horizon.     |
    +----------------+---------------------------+------------------------------+
    | ``ccov``       | ``2.0 / (N**2 + 6.0)``    | Covariance matrix learning   |
    |                |                           | rate.                        |
    +----------------+---------------------------+------------------------------+
    | ``ccovn``      | ``0.4 / (N**1.6 + 1.0)``  | Covariance matrix negative   |
    |                |                           | learning rate.               |
    +----------------+---------------------------+------------------------------+
    | ``cconst``     | ``1.0 / (N + 2.0)``       | Constraint vectors learning  |
    |                |                           | rate.                        |
    +----------------+---------------------------+------------------------------+
    | ``beta``       | ``0.1 / (lambda_ * (N +   | Covariance matrix learning   |
    |                |   2.0))``                 | rate for constraints.        |
    |                |                           |                              |
    +----------------+---------------------------+------------------------------+
    | ``pthresh``    | ``0.44``                  | Threshold success rate.      |
    +----------------+---------------------------+------------------------------+
    .. [Igel2007] Igel, Hansen and Roth. Covariance matrix adaptation for
       multi-objective optimization. 2007
    .. [Arnold2010] Arnold and Hansen. Active covariance matrix adaptation for
       the (1+1)-CMA-ES. 2010.
    .. [Hansen2011] Hansen. A CMA-ES for Mixed-Integer Nonlinear Optimization.
       Research Report] RR-7751, INRIA. 2011
    .. [Arnold2012] Arnold and Hansen. A (1+1)-CMA-ES for Constrained Optimisation.
       2012
    """

    def __init__(self, parent, sigma, steps, **kargs):
        self.parent = parent
        self.sigma = sigma
        self.dim = len(self.parent)
        self.A = numpy.identity(self.dim)
        self.invA = numpy.identity(self.dim)
        self.condition_number = numpy.linalg.cond(self.A)
        self.pc = numpy.zeros(self.dim)
        self.params = kargs.copy()
        self.cc = self.params.get('cc', 2.0 / (self.dim + 2.0))
        self.ccovp = self.params.get('ccovp', 2.0 / (self.dim ** 2 + 6.0))
        self.ccovn = self.params.get('ccovn', 0.4 / (self.dim ** 1.6 + 1.0))
        self.cconst = self.params.get('cconst', 1.0 / (self.dim + 2.0))
        self.pthresh = self.params.get('pthresh', 0.44)
        self.lambda_ = self.params.get('lambda_', 1)
        self.psucc = self.ptarg
        self.S_int = numpy.array(steps)
        self.i_I_R = numpy.flatnonzero(2 * self.sigma * numpy.diag(self.A) ** 0.5 < self.S_int)
        self.constraint_vecs = None
        self.ancestors_fitness = list()

    @property
    def lambda_(self):
        return self._lambda

    @lambda_.setter
    def lambda_(self, value):
        self._lambda = value
        self._compute_lambda_parameters()

    def _compute_lambda_parameters(self):
        """Computes the parameters depending on :math:`\\lambda`. It needs to
        be called again if :math:`\\lambda` changes during evolution.
        """
        self.d = self.params.get('d', 1.0 + self.dim / (2.0 * self.lambda_))
        self.ptarg = self.params.get('ptarg', 1.0 / (5 + numpy.sqrt(self.lambda_) / 2.0))
        self.cp = self.params.get('cp', self.ptarg * self.lambda_ / (2 + self.ptarg * self.lambda_))
        self.beta = self.params.get('beta', 0.1 / (self.lambda_ * (self.dim + 2.0)))

    def generate(self, ind_init):
        """Generate a population of :math:`\\lambda` individuals of type
        *ind_init* from the current strategy.
        :param ind_init: A function object that is able to initialize an
                         individual from a list.
        :returns: A list of individuals.
        """
        z = numpy.random.standard_normal((self.lambda_, self.dim))
        y = numpy.dot(self.A, z.T).T
        x = self.parent + self.sigma * y + self.S_int * self._integer_mutation()
        if any(self.S_int > 0):
            round_values = numpy.tile(self.S_int > 0, (self.lambda_, 1))
            steps = numpy.tile(self.S_int, (self.lambda_, 1))
            x[round_values] = steps[round_values] * numpy.around(x[round_values] / steps[round_values])
        population = list(map(ind_init, x))
        for ind, yi, zi in zip(population, y, z):
            ind._y = yi
            ind._z = zi
        return population

    def _integer_mutation(self):
        n_I_R = self.i_I_R.shape[0]
        if n_I_R == 0:
            return numpy.zeros((self.lambda_, self.dim))
        elif n_I_R == self.dim:
            p = self.lambda_ / 2.0 / self.lambda_
        else:
            p = min(self.lambda_ / 2.0, self.lambda_ / 10.0 + n_I_R / self.dim) / self.lambda_
        Rp = numpy.zeros((self.lambda_, self.dim))
        Rpp = numpy.zeros((self.lambda_, self.dim))
        for i, j in zip(range(self.lambda_), cycle(self.i_I_R)):
            if numpy.random.rand() < p:
                Rp[i, j] = 1
                Rpp[i, j] = numpy.random.geometric(p=0.7 ** (1.0 / n_I_R)) - 1
        I_pm1 = (-1) ** numpy.random.randint(0, 2, (self.lambda_, self.dim))
        R_int = I_pm1 * (Rp + Rpp)
        return R_int

    def _rank1update(self, individual, p_succ):
        update_cov = False
        self.psucc = (1 - self.cp) * self.psucc + self.cp * p_succ
        if not hasattr(self.parent, 'fitness') or self.parent.fitness <= individual.fitness:
            self.parent = copy.deepcopy(individual)
            self.ancestors_fitness.append(copy.deepcopy(individual.fitness))
            if len(self.ancestors_fitness) > 5:
                self.ancestors_fitness.pop()
            if self.psucc < self.pthresh or numpy.allclose(self.pc, 0):
                self.pc = (1 - self.cc) * self.pc + numpy.sqrt(self.cc * (2 - self.cc)) * individual._y
                a = numpy.sqrt(1 - self.ccovp)
                w = numpy.dot(self.invA, self.pc)
                w_norm_sqrd = numpy.linalg.norm(w) ** 2
                b = numpy.sqrt(1 - self.ccovp) / w_norm_sqrd * (numpy.sqrt(1 + self.ccovp / (1 - self.ccovp) * w_norm_sqrd) - 1)
            else:
                self.pc = (1 - self.cc) * self.pc
                d = self.ccovp * (1 + self.cc * (2 - self.cc))
                a = numpy.sqrt(1 - d)
                w = numpy.dot(self.invA, self.pc)
                w_norm_sqrd = numpy.linalg.norm(w) ** 2
                b = numpy.sqrt(1 - d) * (numpy.sqrt(1 + self.ccovp * w_norm_sqrd / (1 - d)) - 1) / w_norm_sqrd
            update_cov = True
        elif len(self.ancestors_fitness) >= 5 and individual.fitness < self.ancestors_fitness[0] and (self.psucc < self.pthresh):
            w = individual._z
            w_norm_sqrd = numpy.linalg.norm(w) ** 2
            if 1 < self.ccovn * (2 * w_norm_sqrd - 1):
                ccovn = 1 / (2 * w_norm_sqrd - 1)
            else:
                ccovn = self.ccovn
            a = numpy.sqrt(1 + ccovn)
            b = numpy.sqrt(1 + ccovn) / w_norm_sqrd * (numpy.sqrt(1 - ccovn / (1 + ccovn) * w_norm_sqrd) - 1)
            update_cov = True
        if update_cov:
            self.A = self.A * a + b * numpy.outer(numpy.dot(self.A, w), w)
            self.invA = 1 / a * self.invA - b / (a ** 2 + a * b * w_norm_sqrd) * numpy.dot(self.invA, numpy.outer(w, w))
        self.sigma = self.sigma * numpy.exp(1.0 / self.d * ((self.psucc - self.ptarg) / (1.0 - self.ptarg)))

    def _infeasible_update(self, individual):
        if not hasattr(individual.fitness, 'constraint_violation'):
            return
        if self.constraint_vecs is None:
            shape = (len(individual.fitness.constraint_violation), self.dim)
            self.constraint_vecs = numpy.zeros(shape)
        for i in range(self.constraint_vecs.shape[0]):
            if individual.fitness.constraint_violation[i]:
                self.constraint_vecs[i] = (1 - self.cconst) * self.constraint_vecs[i] + self.cconst * individual._y
        W = numpy.dot(self.invA, self.constraint_vecs.T).T
        constraint_violation = numpy.sum(individual.fitness.constraint_violation)
        A_prime = self.A - self.beta / constraint_violation * numpy.sum(list((numpy.outer(self.constraint_vecs[i], W[i]) / numpy.dot(W[i], W[i]) for i in range(self.constraint_vecs.shape[0]) if individual.fitness.constraint_violation[i])), axis=0)
        try:
            self.invA = numpy.linalg.inv(A_prime)
        except numpy.linalg.LinAlgError:
            warnings.warn('Singular matrix inversion, invalid update in CMA-ES ignored', RuntimeWarning)
        else:
            self.A = A_prime

    def update(self, population):
        """Update the current covariance matrix strategy from the *population*.
        :param population: A list of individuals from which to update the
                           parameters.
        """
        valid_population = [ind for ind in population if ind.fitness.valid]
        invalid_population = [ind for ind in population if not ind.fitness.valid]
        if len(valid_population) > 0:
            valid_population.sort(key=lambda ind: ind.fitness, reverse=True)
            if not hasattr(self.parent, 'fitness'):
                lambda_succ = len(valid_population)
            else:
                lambda_succ = sum((self.parent.fitness <= ind.fitness for ind in valid_population))
            self._rank1update(valid_population[0], float(lambda_succ) / len(valid_population))
        if len(invalid_population) > 0:
            for ind in invalid_population:
                self._infeasible_update(ind)
        self.condition_number = numpy.linalg.cond(self.A)
        C = numpy.dot(self.A, self.A.T)
        self.i_I_R = numpy.flatnonzero(2 * self.sigma * numpy.diag(C) ** 0.5 < self.S_int)