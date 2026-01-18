import abc
from pyomo.common.dependencies import attempt_import, numpy as np, numpy_available
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
class CyIpoptNLP(CyIpoptProblemInterface):

    def __init__(self, nlp, intermediate_callback=None, halt_on_evaluation_error=None):
        """This class provides a CyIpoptProblemInterface for use
        with the CyIpoptSolver class that can take in an NLP
        as long as it provides vectors as numpy ndarrays and
        matrices as scipy.sparse.coo_matrix objects. This class
        provides the interface between AmplNLP or PyomoNLP objects
        and the CyIpoptSolver
        """
        self._nlp = nlp
        self._intermediate_callback = intermediate_callback
        cyipopt_has_eval_error = cyipopt_available and hasattr(cyipopt, 'CyIpoptEvaluationError')
        if halt_on_evaluation_error is None:
            self._halt_on_evaluation_error = not cyipopt_has_eval_error
        elif not halt_on_evaluation_error and (not cyipopt_has_eval_error):
            raise ValueError('halt_on_evaluation_error=False is only supported for cyipopt >= 1.3.0')
        else:
            self._halt_on_evaluation_error = halt_on_evaluation_error
        x = nlp.init_primals()
        y = nlp.init_duals()
        if np.any(np.isnan(y)):
            y.fill(1.0)
        self._cached_x = x.copy()
        self._cached_y = y.copy()
        self._cached_obj_factor = 1.0
        nlp.set_primals(self._cached_x)
        nlp.set_duals(self._cached_y)
        self._jac_g = nlp.evaluate_jacobian()
        try:
            self._hess_lag = nlp.evaluate_hessian_lag()
            self._hess_lower_mask = self._hess_lag.row >= self._hess_lag.col
            self._hessian_available = True
        except (AttributeError, NotImplementedError):
            self._hessian_available = False
            self._hess_lag = None
            self._hess_lower_mask = None
        super(CyIpoptNLP, self).__init__()

    def _set_primals_if_necessary(self, x):
        if not np.array_equal(x, self._cached_x):
            self._nlp.set_primals(x)
            self._cached_x = x.copy()

    def _set_duals_if_necessary(self, y):
        if not np.array_equal(y, self._cached_y):
            self._nlp.set_duals(y)
            self._cached_y = y.copy()

    def _set_obj_factor_if_necessary(self, obj_factor):
        if obj_factor != self._cached_obj_factor:
            self._nlp.set_obj_factor(obj_factor)
            self._cached_obj_factor = obj_factor

    def x_init(self):
        return self._nlp.init_primals()

    def x_lb(self):
        return self._nlp.primals_lb()

    def x_ub(self):
        return self._nlp.primals_ub()

    def g_lb(self):
        return self._nlp.constraints_lb()

    def g_ub(self):
        return self._nlp.constraints_ub()

    def scaling_factors(self):
        obj_scaling = self._nlp.get_obj_scaling()
        x_scaling = self._nlp.get_primals_scaling()
        g_scaling = self._nlp.get_constraints_scaling()
        return (obj_scaling, x_scaling, g_scaling)

    def objective(self, x):
        try:
            self._set_primals_if_necessary(x)
            return self._nlp.evaluate_objective()
        except PyNumeroEvaluationError:
            if self._halt_on_evaluation_error:
                raise
            else:
                raise cyipopt.CyIpoptEvaluationError('Error in objective function evaluation')

    def gradient(self, x):
        try:
            self._set_primals_if_necessary(x)
            return self._nlp.evaluate_grad_objective()
        except PyNumeroEvaluationError:
            if self._halt_on_evaluation_error:
                raise
            else:
                raise cyipopt.CyIpoptEvaluationError('Error in objective gradient evaluation')

    def constraints(self, x):
        try:
            self._set_primals_if_necessary(x)
            return self._nlp.evaluate_constraints()
        except PyNumeroEvaluationError:
            if self._halt_on_evaluation_error:
                raise
            else:
                raise cyipopt.CyIpoptEvaluationError('Error in constraint evaluation')

    def jacobianstructure(self):
        return (self._jac_g.row, self._jac_g.col)

    def jacobian(self, x):
        try:
            self._set_primals_if_necessary(x)
            self._nlp.evaluate_jacobian(out=self._jac_g)
            return self._jac_g.data
        except PyNumeroEvaluationError:
            if self._halt_on_evaluation_error:
                raise
            else:
                raise cyipopt.CyIpoptEvaluationError('Error in constraint Jacobian evaluation')

    def hessianstructure(self):
        if not self._hessian_available:
            return (np.zeros(0), np.zeros(0))
        row = np.compress(self._hess_lower_mask, self._hess_lag.row)
        col = np.compress(self._hess_lower_mask, self._hess_lag.col)
        return (row, col)

    def hessian(self, x, y, obj_factor):
        if not self._hessian_available:
            raise ValueError('Hessian requested, but not supported by the NLP')
        try:
            self._set_primals_if_necessary(x)
            self._set_duals_if_necessary(y)
            self._set_obj_factor_if_necessary(obj_factor)
            self._nlp.evaluate_hessian_lag(out=self._hess_lag)
            data = np.compress(self._hess_lower_mask, self._hess_lag.data)
            return data
        except PyNumeroEvaluationError:
            if self._halt_on_evaluation_error:
                raise
            else:
                raise cyipopt.CyIpoptEvaluationError('Error in Lagrangian Hessian evaluation')

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        if self._intermediate_callback is not None:
            return self._intermediate_callback(self._nlp, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials)
        return True