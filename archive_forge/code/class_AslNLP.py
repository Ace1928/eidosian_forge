from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
class AslNLP(ExtendedNLP):

    def __init__(self, nl_file):
        """
        Base class for NLP classes based on the Ampl Solver Library and
        NL files.

        Parameters
        ----------
        nl_file : string
            filename of the NL-file containing the model
        """
        super(AslNLP, self).__init__()
        self._nl_file = nl_file
        self._asl = _asl.AmplInterface(self._nl_file)
        self._collect_nlp_structure()
        self._primals = self._init_primals.copy()
        self._duals_full = self._init_duals_full.copy()
        self._duals_eq = self._init_duals_eq.copy()
        self._duals_ineq = self._init_duals_ineq.copy()
        self._obj_factor = 1.0
        self._cached_objective = None
        self._cached_grad_objective = self.create_new_vector('primals')
        self._cached_con_full = np.zeros(self._n_con_full, dtype=np.float64)
        self._cached_jac_full = coo_matrix((np.zeros(self._nnz_jac_full, dtype=np.float64), (self._irows_jac_full, self._jcols_jac_full)), shape=(self._n_con_full, self._n_primals))
        self._cached_jac_eq = coo_matrix((np.zeros(self._nnz_jac_eq, dtype=np.float64), (self._irows_jac_eq, self._jcols_jac_eq)), shape=(self._n_con_eq, self._n_primals))
        self._cached_jac_ineq = coo_matrix((np.zeros(self._nnz_jac_ineq), (self._irows_jac_ineq, self._jcols_jac_ineq)), shape=(self._n_con_ineq, self._n_primals))
        self._cached_hessian_lag = coo_matrix((np.zeros(self._nnz_hessian_lag, dtype=np.float64), (self._irows_hess, self._jcols_hess)), shape=(self._n_primals, self._n_primals))
        self._invalidate_primals_cache()
        self._invalidate_duals_cache()
        self._invalidate_obj_factor_cache()

    def _invalidate_primals_cache(self):
        self._objective_is_cached = False
        self._grad_objective_is_cached = False
        self._con_full_is_cached = False
        self._jac_full_is_cached = False
        self._hessian_lag_is_cached = False

    def _invalidate_duals_cache(self):
        self._hessian_lag_is_cached = False

    def _invalidate_obj_factor_cache(self):
        self._hessian_lag_is_cached = False

    def _collect_nlp_structure(self):
        """
        Collect characteristics of the NLP from the ASL interface
        """
        self._n_primals = self._asl.get_n_vars()
        self._n_con_full = self._asl.get_n_constraints()
        self._nnz_jac_full = self._asl.get_nnz_jac_g()
        self._nnz_hess_lag_lower = self._asl.get_nnz_hessian_lag()
        self._init_primals = np.zeros(self._n_primals, dtype=np.float64)
        self._init_duals_full = np.zeros(self._n_con_full, dtype=np.float64)
        self._asl.get_init_x(self._init_primals)
        self._asl.get_init_multipliers(self._init_duals_full)
        self._init_primals.flags.writeable = False
        self._init_duals_full.flags.writeable = False
        self._primals_lb = np.zeros(self._n_primals, dtype=np.float64)
        self._primals_ub = np.zeros(self._n_primals, dtype=np.float64)
        self._asl.get_x_lower_bounds(self._primals_lb)
        self._asl.get_x_upper_bounds(self._primals_ub)
        self._primals_lb.flags.writeable = False
        self._primals_ub.flags.writeable = False
        self._con_full_lb = np.zeros(self._n_con_full, dtype=np.float64)
        self._con_full_ub = np.zeros(self._n_con_full, dtype=np.float64)
        self._asl.get_g_lower_bounds(self._con_full_lb)
        self._asl.get_g_upper_bounds(self._con_full_ub)
        bounds_difference = self._primals_ub - self._primals_lb
        if np.any(bounds_difference < 0):
            print(np.where(bounds_difference < 0))
            raise RuntimeError('Some variables have lower bounds that are greater than the upper bounds.')
        self._build_constraint_maps()
        self._con_ineq_lb = np.compress(self._con_full_ineq_mask, self._con_full_lb)
        self._con_ineq_ub = np.compress(self._con_full_ineq_mask, self._con_full_ub)
        self._con_ineq_lb.flags.writeable = False
        self._con_ineq_ub.flags.writeable = False
        self._init_duals_eq = np.compress(self._con_full_eq_mask, self._init_duals_full)
        self._init_duals_ineq = np.compress(self._con_full_ineq_mask, self._init_duals_full)
        self._init_duals_eq.flags.writeable = False
        self._init_duals_ineq.flags.writeable = False
        self._con_full_rhs = self._con_full_ub.copy()
        self._con_full_rhs[~self._con_full_eq_mask] = 0.0
        self._con_full_lb[self._con_full_eq_mask] = 0.0
        self._con_full_ub[self._con_full_eq_mask] = 0.0
        self._con_full_lb.flags.writeable = False
        self._con_full_ub.flags.writeable = False
        self._n_con_eq = len(self._con_eq_full_map)
        self._n_con_ineq = len(self._con_ineq_full_map)
        self._irows_jac_full = np.zeros(self._nnz_jac_full, dtype=np.intc)
        self._jcols_jac_full = np.zeros(self._nnz_jac_full, dtype=np.intc)
        self._asl.struct_jac_g(self._irows_jac_full, self._jcols_jac_full)
        self._irows_jac_full -= 1
        self._jcols_jac_full -= 1
        self._irows_jac_full.flags.writeable = False
        self._jcols_jac_full.flags.writeable = False
        self._nz_con_full_eq_mask = np.isin(self._irows_jac_full, self._con_eq_full_map)
        self._nz_con_full_ineq_mask = np.logical_not(self._nz_con_full_eq_mask)
        self._irows_jac_eq = np.compress(self._nz_con_full_eq_mask, self._irows_jac_full)
        self._jcols_jac_eq = np.compress(self._nz_con_full_eq_mask, self._jcols_jac_full)
        self._irows_jac_ineq = np.compress(self._nz_con_full_ineq_mask, self._irows_jac_full)
        self._jcols_jac_ineq = np.compress(self._nz_con_full_ineq_mask, self._jcols_jac_full)
        self._nnz_jac_eq = len(self._irows_jac_eq)
        self._nnz_jac_ineq = len(self._irows_jac_ineq)
        self._con_full_eq_map = full_eq_map = {self._con_eq_full_map[i]: i for i in range(self._n_con_eq)}
        for i, v in enumerate(self._irows_jac_eq):
            self._irows_jac_eq[i] = full_eq_map[v]
        self._con_full_ineq_map = full_ineq_map = {self._con_ineq_full_map[i]: i for i in range(self._n_con_ineq)}
        for i, v in enumerate(self._irows_jac_ineq):
            self._irows_jac_ineq[i] = full_ineq_map[v]
        self._irows_jac_eq.flags.writeable = False
        self._jcols_jac_eq.flags.writeable = False
        self._irows_jac_ineq.flags.writeable = False
        self._jcols_jac_ineq.flags.writeable = False
        self._nnz_jac_eq = len(self._jcols_jac_eq)
        self._nnz_jac_ineq = len(self._jcols_jac_ineq)
        self._irows_hess = np.zeros(self._nnz_hess_lag_lower, dtype=np.intc)
        self._jcols_hess = np.zeros(self._nnz_hess_lag_lower, dtype=np.intc)
        self._asl.struct_hes_lag(self._irows_hess, self._jcols_hess)
        self._irows_hess -= 1
        self._jcols_hess -= 1
        diff = self._irows_hess - self._jcols_hess
        self._lower_hess_mask = np.where(diff != 0)
        lower = self._lower_hess_mask
        self._irows_hess = np.concatenate((self._irows_hess, self._jcols_hess[lower]))
        self._jcols_hess = np.concatenate((self._jcols_hess, self._irows_hess[lower]))
        self._nnz_hessian_lag = self._irows_hess.size
        self._irows_hess.flags.writeable = False
        self._jcols_hess.flags.writeable = False

    def _build_constraint_maps(self):
        """Creates internal maps and masks that convert from the full
        vector of constraints (the vector that includes all equality
        and inequality constraints combined) to separate vectors that
        include the equality and inequality constraints only.
        """
        bounds_difference = self._con_full_ub - self._con_full_lb
        inconsistent_bounds = np.any(bounds_difference < 0.0)
        if inconsistent_bounds:
            raise RuntimeError('Bounds on range constraints found with upper bounds set below the lower bounds.')
        abs_bounds_difference = np.absolute(bounds_difference)
        tolerance_equalities = 1e-08
        self._con_full_eq_mask = abs_bounds_difference < tolerance_equalities
        self._con_eq_full_map = self._con_full_eq_mask.nonzero()[0]
        self._con_full_ineq_mask = abs_bounds_difference >= tolerance_equalities
        self._con_ineq_full_map = self._con_full_ineq_mask.nonzero()[0]
        self._con_full_eq_mask.flags.writeable = False
        self._con_eq_full_map.flags.writeable = False
        self._con_full_ineq_mask.flags.writeable = False
        self._con_ineq_full_map.flags.writeable = False
        '\n        #TODO: Can we simplify this logic?\n        con_full_fulllb_mask = np.isfinite(self._con_full_lb) * self._con_full_ineq_mask + self._con_full_eq_mask\n        con_fulllb_full_map = con_full_fulllb_mask.nonzero()[0]\n        con_full_fullub_mask = np.isfinite(self._con_full_ub) * self._con_full_ineq_mask + self._con_full_eq_mask\n        con_fullub_full_map = con_full_fullub_mask.nonzero()[0]\n\n        self._ineq_lb_mask = np.isin(self._ineq_g_map, lb_g_map)\n        self._lb_ineq_map = np.where(self._ineq_lb_mask)[0]\n        self._ineq_ub_mask = np.isin(self._ineq_g_map, ub_g_map)\n        self._ub_ineq_map = np.where(self._ineq_ub_mask)[0]\n        self._ineq_lb_mask.flags.writeable = False\n        self._lb_ineq_map.flags.writeable = False\n        self._ineq_ub_mask.flags.writeable = False\n        self._ub_ineq_map.flags.writeable = False\n        '

    def n_primals(self):
        return self._n_primals

    def n_constraints(self):
        return self._n_con_full

    def n_eq_constraints(self):
        return self._n_con_eq

    def n_ineq_constraints(self):
        return self._n_con_ineq

    def nnz_jacobian(self):
        return self._nnz_jac_full

    def nnz_jacobian_eq(self):
        return self._nnz_jac_eq

    def nnz_jacobian_ineq(self):
        return self._nnz_jac_ineq

    def nnz_hessian_lag(self):
        return self._nnz_hessian_lag

    def primals_lb(self):
        return self._primals_lb

    def primals_ub(self):
        return self._primals_ub

    def constraints_lb(self):
        return self._con_full_lb

    def constraints_ub(self):
        return self._con_full_ub

    def ineq_lb(self):
        return self._con_ineq_lb

    def ineq_ub(self):
        return self._con_ineq_ub

    def init_primals(self):
        return self._init_primals

    def init_duals(self):
        return self._init_duals_full

    def init_duals_eq(self):
        return self._init_duals_eq

    def init_duals_ineq(self):
        return self._init_duals_ineq

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
        numpy.ndarray
        """
        if vector_type == 'primals':
            return np.zeros(self.n_primals(), dtype=np.float64)
        elif vector_type == 'constraints' or vector_type == 'duals':
            return np.zeros(self.n_constraints(), dtype=np.float64)
        elif vector_type == 'eq_constraints' or vector_type == 'duals_eq':
            return np.zeros(self.n_eq_constraints(), dtype=np.float64)
        elif vector_type == 'ineq_constraints' or vector_type == 'duals_ineq':
            return np.zeros(self.n_ineq_constraints(), dtype=np.float64)
        else:
            raise RuntimeError('Called create_new_vector with an unknown vector_type')

    def set_primals(self, primals):
        self._invalidate_primals_cache()
        np.copyto(self._primals, primals)

    def get_primals(self):
        return self._primals.copy()

    def set_duals(self, duals):
        self._invalidate_duals_cache()
        np.copyto(self._duals_full, duals)
        np.compress(self._con_full_eq_mask, self._duals_full, out=self._duals_eq)
        np.compress(self._con_full_ineq_mask, self._duals_full, out=self._duals_ineq)

    def get_duals(self):
        return self._duals_full.copy()

    def set_obj_factor(self, obj_factor):
        self._invalidate_obj_factor_cache()
        self._obj_factor = obj_factor

    def get_obj_factor(self):
        return self._obj_factor

    def set_duals_eq(self, duals_eq):
        self._invalidate_duals_cache()
        np.copyto(self._duals_eq, duals_eq)
        self._duals_full[self._con_full_eq_mask] = self._duals_eq

    def get_duals_eq(self):
        return self._duals_eq.copy()

    def set_duals_ineq(self, duals_ineq):
        self._invalidate_duals_cache()
        np.copyto(self._duals_ineq, duals_ineq)
        self._duals_full[self._con_full_ineq_mask] = self._duals_ineq

    def get_duals_ineq(self):
        return self._duals_ineq.copy()

    def get_obj_scaling(self):
        return None

    def get_primals_scaling(self):
        return None

    def get_constraints_scaling(self):
        return None

    def get_eq_constraints_scaling(self):
        constraints_scaling = self.get_constraints_scaling()
        if constraints_scaling is not None:
            return np.compress(self._con_full_eq_mask, constraints_scaling)
        return None

    def get_ineq_constraints_scaling(self):
        constraints_scaling = self.get_constraints_scaling()
        if constraints_scaling is not None:
            return np.compress(self._con_full_ineq_mask, constraints_scaling)
        return None

    def _evaluate_objective_and_cache_if_necessary(self):
        if not self._objective_is_cached:
            self._cached_objective = self._asl.eval_f(self._primals)
            self._objective_is_cached = True

    def evaluate_objective(self):
        self._evaluate_objective_and_cache_if_necessary()
        return self._cached_objective

    def evaluate_grad_objective(self, out=None):
        if not self._grad_objective_is_cached:
            self._asl.eval_deriv_f(self._primals, self._cached_grad_objective)
            self._grad_objective_is_cached = True
        if out is not None:
            if not isinstance(out, np.ndarray) or out.size != self._n_primals:
                raise RuntimeError('Called evaluate_grad_objective with an invalid "out" argument - should take an ndarray of size {}'.format(self._n_primals))
            np.copyto(out, self._cached_grad_objective)
            return out
        else:
            return self._cached_grad_objective.copy()

    def _evaluate_constraints_and_cache_if_necessary(self):
        if not self._con_full_is_cached:
            self._asl.eval_g(self._primals, self._cached_con_full)
            self._cached_con_full -= self._con_full_rhs
            self._con_full_is_cached = True

    def evaluate_constraints(self, out=None):
        self._evaluate_constraints_and_cache_if_necessary()
        if out is not None:
            if not isinstance(out, np.ndarray) or out.size != self._n_con_full:
                raise RuntimeError('Called evaluate_constraints with an invalid "out" argument - should take an ndarray of size {}'.format(self._n_con_full))
            np.copyto(out, self._cached_con_full)
            return out
        else:
            return self._cached_con_full.copy()

    def evaluate_eq_constraints(self, out=None):
        self._evaluate_constraints_and_cache_if_necessary()
        if out is not None:
            if not isinstance(out, np.ndarray) or out.size != self._n_con_eq:
                raise RuntimeError('Called evaluate_eq_constraints with an invalid "out" argument - should take an ndarray of size {}'.format(self._n_con_eq))
            self._cached_con_full.compress(self._con_full_eq_mask, out=out)
            return out
        else:
            return self._cached_con_full.compress(self._con_full_eq_mask)

    def evaluate_ineq_constraints(self, out=None):
        self._evaluate_constraints_and_cache_if_necessary()
        if out is not None:
            if not isinstance(out, np.ndarray) or out.size != self._n_con_ineq:
                raise RuntimeError('Called evaluate_ineq_constraints with an invalid "out" argument - should take an ndarray of size {}'.format(self._n_con_ineq))
            self._cached_con_full.compress(self._con_full_ineq_mask, out=out)
            return out
        else:
            return self._cached_con_full.compress(self._con_full_ineq_mask)

    def _evaluate_jacobians_and_cache_if_necessary(self):
        if not self._jac_full_is_cached:
            self._asl.eval_jac_g(self._primals, self._cached_jac_full.data)
            self._jac_full_is_cached = True

    def evaluate_jacobian(self, out=None):
        self._evaluate_jacobians_and_cache_if_necessary()
        if out is not None:
            if not isinstance(out, coo_matrix) or out.shape[0] != self._n_con_full or out.shape[1] != self._n_primals or (out.nnz != self._nnz_jac_full):
                raise RuntimeError('evaluate_jacobian called with an "out" argument that is invalid. This should be a coo_matrix with shape=({},{}) and nnz={}'.format(self._n_con_full, self._n_primals, self._nnz_jac_full))
            np.copyto(out.data, self._cached_jac_full.data)
            return out
        else:
            return self._cached_jac_full.copy()

    def evaluate_jacobian_eq(self, out=None):
        self._evaluate_jacobians_and_cache_if_necessary()
        if out is not None:
            if not isinstance(out, coo_matrix) or out.shape[0] != self._n_con_eq or out.shape[1] != self._n_primals or (out.nnz != self._nnz_jac_eq):
                raise RuntimeError('evaluate_jacobian_eq called with an "out" argument that is invalid. This should be a coo_matrix with shape=({},{}) and nnz={}'.format(self._n_con_eq, self._n_primals, self._nnz_jac_eq))
            self._cached_jac_full.data.compress(self._nz_con_full_eq_mask, out=out.data)
            return out
        else:
            self._cached_jac_full.data.compress(self._nz_con_full_eq_mask, out=self._cached_jac_eq.data)
            return self._cached_jac_eq.copy()

    def evaluate_jacobian_ineq(self, out=None):
        self._evaluate_jacobians_and_cache_if_necessary()
        if out is not None:
            if not isinstance(out, coo_matrix) or out.shape[0] != self._n_con_ineq or out.shape[1] != self._n_primals or (out.nnz != self._nnz_jac_ineq):
                raise RuntimeError('evaluate_jacobian_ineq called with an "out" argument that is invalid. This should be a coo_matrix with shape=({},{}) and nnz={}'.format(self._n_con_ineq, self._n_primals, self._nnz_jac_ineq))
            self._cached_jac_full.data.compress(self._nz_con_full_ineq_mask, out=out.data)
            return out
        else:
            self._cached_jac_full.data.compress(self._nz_con_full_ineq_mask, out=self._cached_jac_ineq.data)
            return self._cached_jac_ineq.copy()

    def evaluate_hessian_lag(self, out=None):
        if not self._hessian_lag_is_cached:
            self._evaluate_objective_and_cache_if_necessary()
            self._evaluate_constraints_and_cache_if_necessary()
            data = np.zeros(self._nnz_hess_lag_lower, np.float64)
            self._asl.eval_hes_lag(self._primals, self._duals_full, data, obj_factor=self._obj_factor)
            values = np.concatenate((data, data[self._lower_hess_mask]))
            np.copyto(self._cached_hessian_lag.data, values)
            self._hessian_lag_is_cached = True
        if out is not None:
            if not isinstance(out, coo_matrix) or out.shape[0] != self._n_primals or out.shape[1] != self._n_primals or (out.nnz != self._nnz_hessian_lag):
                raise RuntimeError('evaluate_hessian_lag called with an "out" argument that is invalid. This should be a coo_matrix with shape=({},{}) and nnz={}'.format(self._n_primals, self._n_primals, self._nnz_hessian_lag))
            np.copyto(out.data, self._cached_hessian_lag.data)
            return out
        else:
            return self._cached_hessian_lag.copy()

    def report_solver_status(self, status_code, status_message):
        self._asl.finalize_solution(status_code, status_message, self._primals, self._duals)