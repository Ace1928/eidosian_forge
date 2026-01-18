import abc
from pyomo.common.dependencies import attempt_import, numpy as np, numpy_available
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
class CyIpoptProblemInterface(cyipopt_Problem, metaclass=abc.ABCMeta):
    """Abstract subclass of ``cyipopt.Problem`` defining an object that can be
    used as an interface to CyIpopt. Subclasses must define all methods necessary
    for the CyIpopt solve and must call this class's ``__init__`` method to
    initialize Ipopt's data structures.

    Note that, if "output_file" is provided as an Ipopt option, the log file
    is open until this object (and thus the underlying Ipopt NLP object) is
    deallocated. To force this deallocation, call the ``close()`` method, which
    is defined by ``cyipopt.Problem``.

    """
    _problem_initialized = False

    def __init__(self):
        """Initialize the problem interface

        This method calls ``cyipopt.Problem.__init__``, and *must* be called
        by any subclass's ``__init__`` method. If not, we will segfault when
        we call ``cyipopt.Problem.solve`` from this object.

        """
        if not cyipopt_available:
            raise RuntimeError('cyipopt is required to instantiate CyIpoptProblemInterface')
        xl = self.x_lb()
        xu = self.x_ub()
        gl = self.g_lb()
        gu = self.g_ub()
        nx = len(xl)
        ng = len(gl)
        super(CyIpoptProblemInterface, self).__init__(n=nx, m=ng, lb=xl, ub=xu, cl=gl, cu=gu)
        self._problem_initialized = True

    def solve(self, x, lagrange=None, zl=None, zu=None):
        """Solve a CyIpopt Problem

        Checks whether __init__ has been called before calling
        cyipopt.Problem.solve

        """
        lagrange = [] if lagrange is None else lagrange
        zl = [] if zl is None else zl
        zu = [] if zu is None else zu
        if not self._problem_initialized:
            raise RuntimeError('Attempting to call cyipopt.Problem.solve when cyipopt.Problem.__init__ has not been called. This can happen if a subclass of CyIpoptProblemInterface overrides __init__ without calling CyIpoptProblemInterface.__init__ or setting the CyIpoptProblemInterface._problem_initialized flag.')
        return super(CyIpoptProblemInterface, self).solve(x, lagrange=lagrange, zl=zl, zu=zu)

    @abc.abstractmethod
    def x_init(self):
        """Return the initial values for x as a numpy ndarray"""
        pass

    @abc.abstractmethod
    def x_lb(self):
        """Return the lower bounds on x as a numpy ndarray"""
        pass

    @abc.abstractmethod
    def x_ub(self):
        """Return the upper bounds on x as a numpy ndarray"""
        pass

    @abc.abstractmethod
    def g_lb(self):
        """Return the lower bounds on the constraints as a numpy ndarray"""
        pass

    @abc.abstractmethod
    def g_ub(self):
        """Return the upper bounds on the constraints as a numpy ndarray"""
        pass

    @abc.abstractmethod
    def scaling_factors(self):
        """Return the values for scaling factors as a tuple
        (objective_scaling, x_scaling, g_scaling). Return None
        if the scaling factors are to be ignored
        """
        pass

    @abc.abstractmethod
    def objective(self, x):
        """Return the value of the objective
        function evaluated at x
        """
        pass

    @abc.abstractmethod
    def gradient(self, x):
        """Return the gradient of the objective
        function evaluated at x as a numpy ndarray
        """
        pass

    @abc.abstractmethod
    def constraints(self, x):
        """Return the residuals of the constraints
        evaluated at x as a numpy ndarray
        """
        pass

    @abc.abstractmethod
    def jacobianstructure(self):
        """Return the structure of the jacobian
        in coordinate format. That is, return (rows,cols)
        where rows and cols are both numpy ndarray
        objects that contain the row and column indices
        for each of the nonzeros in the jacobian.
        """
        pass

    @abc.abstractmethod
    def jacobian(self, x):
        """Return the values for the jacobian evaluated at x
        as a numpy ndarray of nonzero values corresponding
        to the rows and columns specified in the jacobianstructure
        """
        pass

    @abc.abstractmethod
    def hessianstructure(self):
        """Return the structure of the hessian
        in coordinate format. That is, return (rows,cols)
        where rows and cols are both numpy ndarray
        objects that contain the row and column indices
        for each of the nonzeros in the hessian.
        Note: return ONLY the lower diagonal of this symmetric matrix.
        """
        pass

    @abc.abstractmethod
    def hessian(self, x, y, obj_factor):
        """Return the values for the hessian evaluated at x
        as a numpy ndarray of nonzero values corresponding
        to the rows and columns specified in the
        hessianstructure method.
        Note: return ONLY the lower diagonal of this symmetric matrix.
        """
        pass

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        """Callback that can be used to examine or report intermediate
        results. This method is called each iteration
        """
        pass