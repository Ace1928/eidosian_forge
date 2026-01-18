from abc import ABCMeta, abstractmethod
from pyomo.contrib.pynumero.interfaces import pyomo_nlp, ampl_nlp
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
import numpy as np
import scipy.sparse
from pyomo.common.timing import HierarchicalTimer
class BaseInteriorPointInterface(object, metaclass=ABCMeta):

    @abstractmethod
    def n_primals(self):
        pass

    @abstractmethod
    def nnz_hessian_lag(self):
        pass

    @abstractmethod
    def primals_lb(self):
        pass

    @abstractmethod
    def primals_ub(self):
        pass

    @abstractmethod
    def init_primals(self):
        pass

    @abstractmethod
    def set_primals(self, primals):
        pass

    @abstractmethod
    def get_primals(self):
        pass

    @abstractmethod
    def get_obj_factor(self):
        pass

    @abstractmethod
    def set_obj_factor(self, obj_factor):
        pass

    @abstractmethod
    def evaluate_objective(self):
        pass

    @abstractmethod
    def evaluate_grad_objective(self):
        pass

    @abstractmethod
    def n_eq_constraints(self):
        pass

    @abstractmethod
    def n_ineq_constraints(self):
        pass

    @abstractmethod
    def nnz_jacobian_eq(self):
        pass

    @abstractmethod
    def nnz_jacobian_ineq(self):
        pass

    @abstractmethod
    def ineq_lb(self):
        pass

    @abstractmethod
    def ineq_ub(self):
        pass

    @abstractmethod
    def init_duals_eq(self):
        pass

    @abstractmethod
    def init_duals_ineq(self):
        pass

    @abstractmethod
    def set_duals_eq(self, duals_eq):
        pass

    @abstractmethod
    def set_duals_ineq(self, duals_ineq):
        pass

    @abstractmethod
    def get_duals_eq(self):
        pass

    @abstractmethod
    def get_duals_ineq(self):
        pass

    @abstractmethod
    def evaluate_eq_constraints(self):
        pass

    @abstractmethod
    def evaluate_ineq_constraints(self):
        pass

    @abstractmethod
    def evaluate_jacobian_eq(self):
        pass

    @abstractmethod
    def evaluate_jacobian_ineq(self):
        pass

    @abstractmethod
    def init_slacks(self):
        pass

    @abstractmethod
    def init_duals_primals_lb(self):
        pass

    @abstractmethod
    def init_duals_primals_ub(self):
        pass

    @abstractmethod
    def init_duals_slacks_lb(self):
        pass

    @abstractmethod
    def init_duals_slacks_ub(self):
        pass

    @abstractmethod
    def set_slacks(self, slacks):
        pass

    @abstractmethod
    def set_duals_primals_lb(self, duals):
        pass

    @abstractmethod
    def set_duals_primals_ub(self, duals):
        pass

    @abstractmethod
    def set_duals_slacks_lb(self, duals):
        pass

    @abstractmethod
    def set_duals_slacks_ub(self, duals):
        pass

    @abstractmethod
    def get_slacks(self):
        pass

    @abstractmethod
    def get_duals_primals_lb(self):
        pass

    @abstractmethod
    def get_duals_primals_ub(self):
        pass

    @abstractmethod
    def get_duals_slacks_lb(self):
        pass

    @abstractmethod
    def get_duals_slacks_ub(self):
        pass

    @abstractmethod
    def set_barrier_parameter(self, barrier):
        pass

    @abstractmethod
    def evaluate_primal_dual_kkt_matrix(self, timer=None):
        pass

    @abstractmethod
    def evaluate_primal_dual_kkt_rhs(self, timer=None):
        pass

    @abstractmethod
    def set_primal_dual_kkt_solution(self, sol):
        pass

    @abstractmethod
    def get_delta_primals(self):
        pass

    @abstractmethod
    def get_delta_slacks(self):
        pass

    @abstractmethod
    def get_delta_duals_eq(self):
        pass

    @abstractmethod
    def get_delta_duals_ineq(self):
        pass

    @abstractmethod
    def get_delta_duals_primals_lb(self):
        pass

    @abstractmethod
    def get_delta_duals_primals_ub(self):
        pass

    @abstractmethod
    def get_delta_duals_slacks_lb(self):
        pass

    @abstractmethod
    def get_delta_duals_slacks_ub(self):
        pass

    def regularize_equality_gradient(self, kkt, coef, copy_kkt=True):
        raise RuntimeError('Equality gradient regularization is necessary but no function has been implemented for doing so.')

    def regularize_hessian(self, kkt, coef, copy_kkt=True):
        raise RuntimeError('Hessian of Lagrangian regularization is necessary but no function has been implemented for doing so.')