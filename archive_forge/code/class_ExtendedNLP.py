import abc
class ExtendedNLP(NLP, metaclass=abc.ABCMeta):
    """This interface extends the NLP interface to support a presentation
    of the problem that separates equality and inequality constraints
    """

    def __init__(self):
        super(ExtendedNLP, self).__init__()
        pass

    @abc.abstractmethod
    def n_eq_constraints(self):
        """
        Returns number of equality constraints
        """
        pass

    @abc.abstractmethod
    def n_ineq_constraints(self):
        """
        Returns number of inequality constraints
        """
        pass

    @abc.abstractmethod
    def nnz_jacobian_eq(self):
        """
        Returns number of nonzero values in jacobian of equality constraints
        """
        pass

    @abc.abstractmethod
    def nnz_jacobian_ineq(self):
        """
        Returns number of nonzero values in jacobian of inequality constraints
        """
        pass

    @abc.abstractmethod
    def ineq_lb(self):
        """
        Returns vector of lower bounds for inequality constraints

        Returns
        -------
        vector-like

        """
        pass

    @abc.abstractmethod
    def ineq_ub(self):
        """
        Returns vector of upper bounds for inequality constraints

        Returns
        -------
        vector-like

        """
        pass

    @abc.abstractmethod
    def init_duals_eq(self):
        """
        Returns vector with initial values for the dual variables of the
        equality constraints
        """
        pass

    @abc.abstractmethod
    def init_duals_ineq(self):
        """
        Returns vector with initial values for the dual variables of the
        inequality constraints
        """
        pass

    @abc.abstractmethod
    def create_new_vector(self, vector_type):
        """
        Creates a vector of the appropriate length and structure as
        requested

        Parameters
        ----------
        vector_type: {'primals', 'constraints', 'eq_constraints', 'ineq_constraints',
                      'duals', 'duals_eq', 'duals_ineq'}
            String identifying the appropriate  vector  to create.

        Returns
        -------
        vector-like
        """
        pass

    @abc.abstractmethod
    def set_duals_eq(self, duals_eq):
        """Set the value of the dual variables for the equality constraints
        to be used in calls to the evaluation methods (hessian_lag)

        Parameters
        ----------
        duals_eq: vector_like
            Vector with the values of dual variables for the equality constraints
        """
        pass

    @abc.abstractmethod
    def get_duals_eq(self):
        """Get a copy of the values of the dual variables of the equality
        constraints as provided in set_duals_eq. These are the values
        that will be used in calls to the evaluation methods.
        """
        pass

    @abc.abstractmethod
    def set_duals_ineq(self, duals_ineq):
        """Set the value of the dual variables for the inequality constraints
        to be used in calls to the evaluation methods (hessian_lag)

        Parameters
        ----------
        duals_ineq: vector_like
            Vector with the values of dual variables for the inequality constraints
        """
        pass

    @abc.abstractmethod
    def get_duals_ineq(self):
        """Get a copy of the values of the dual variables of the inequality
        constraints as provided in set_duals_eq. These are the values
        that will be used in calls to the evaluation methods.
        """
        pass

    @abc.abstractmethod
    def get_eq_constraints_scaling(self):
        """Return the desired scaling factors to use for the
        for the equality constraints. None indicates no scaling.
        This indicates potential scaling for the model, but the
        evaluation methods should return *unscaled* values

        Returns
        -------
        array-like or None
        """
        pass

    @abc.abstractmethod
    def get_ineq_constraints_scaling(self):
        """Return the desired scaling factors to use for the
        for the inequality constraints. None indicates no scaling.
        This indicates potential scaling for the model, but the
        evaluation methods should return *unscaled* values

        Returns
        -------
        array-like or None
        """
        pass

    @abc.abstractmethod
    def evaluate_eq_constraints(self, out=None):
        """Returns the values for the equality constraints evaluated at
        the values given for the primal variales in set_primals

        Parameters
        ----------
        out: array_like, optional
            Output array. Its type is preserved and it
            must be of the right shape to hold the output.

        Returns
        -------
        vector_like
        """
        pass

    @abc.abstractmethod
    def evaluate_ineq_constraints(self, out=None):
        """Returns the values of the inequality constraints evaluated at
        the values given for the primal variables in set_primals

        Parameters
        ----------
        out : array_like, optional
            Output array. Its type is preserved and it
            must be of the right shape to hold the output.

        Returns
        -------
        vector_like
        """
        pass

    @abc.abstractmethod
    def evaluate_jacobian_eq(self, out=None):
        """Returns the Jacobian of the equality constraints evaluated
        at the values given for the primal variables in set_primals

        Parameters
        ----------
        out : matrix_like (e.g., coo_matrix), optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        matrix_like
        """
        pass

    @abc.abstractmethod
    def evaluate_jacobian_ineq(self, out=None):
        """Returns the Jacobian of the inequality constraints evaluated
        at the values given for the primal variables in set_primals

        Parameters
        ----------
        out : matrix_like (e.g., coo_matrix), optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        matrix_like
        """
        pass