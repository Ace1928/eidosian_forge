from pyomo.contrib.pynumero.interfaces.utils import (
import numpy as np
import logging
import time
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus
from pyomo.common.timing import HierarchicalTimer
import enum
class InteriorPointSolver(object):
    """
    Class for creating interior point solvers with different options
    """

    def __init__(self, linear_solver, max_iter=100, tol=1e-08, linear_solver_log_filename=None, max_reallocation_iterations=5, reallocation_factor=2):
        self.linear_solver = linear_solver
        self.max_iter = max_iter
        self.tol = tol
        self.linear_solver_log_filename = linear_solver_log_filename
        self.max_reallocation_iterations = max_reallocation_iterations
        self.reallocation_factor = reallocation_factor
        self.base_eq_reg_coef = -1e-08
        self._barrier_parameter = 0.1
        self._minimum_barrier_parameter = 1e-09
        self.hess_reg_coef = 0.0001
        self.max_reg_iter = 6
        self.reg_factor_increase = 100
        self.logger = logging.getLogger('interior_point')
        self._iter = 0
        self.factorization_context = FactorizationContext(self.logger)
        if linear_solver_log_filename:
            with open(linear_solver_log_filename, 'w'):
                pass
        self.linear_solver_logger = self.linear_solver.getLogger()
        self.linear_solve_context = LinearSolveContext(self.logger, self.linear_solver_logger, self.linear_solver_log_filename)

    def update_barrier_parameter(self):
        self._barrier_parameter = max(self._minimum_barrier_parameter, min(0.5 * self._barrier_parameter, self._barrier_parameter ** 1.5))

    def set_linear_solver(self, linear_solver):
        """This method exists to hopefully make it easy to try the same IP
        algorithm with different linear solvers.
        Subclasses may have linear-solver specific methods, in which case
        this should not be called.

        Hopefully the linear solver interface can be standardized such that
        this is not a problem. (Need a generalized method for set_options)
        """
        self.linear_solver = linear_solver

    def set_interface(self, interface):
        self.interface = interface

    def solve(self, interface, timer=None, report_timing=False):
        """
        Parameters
        ----------
        interface: pyomo.contrib.interior_point.interface.BaseInteriorPointInterface
            The interior point interface. This object handles the function evaluation,
            building the KKT matrix, and building the KKT right hand side.
        timer: HierarchicalTimer
        report_timing: bool
        """
        linear_solver = self.linear_solver
        max_iter = self.max_iter
        tol = self.tol
        if timer is None:
            timer = HierarchicalTimer()
        timer.start('IP solve')
        timer.start('init')
        self._barrier_parameter = 0.1
        self.set_interface(interface)
        t0 = time.time()
        primals = interface.init_primals().copy()
        slacks = interface.init_slacks().copy()
        duals_eq = interface.init_duals_eq().copy()
        duals_ineq = interface.init_duals_ineq().copy()
        duals_primals_lb = interface.init_duals_primals_lb().copy()
        duals_primals_ub = interface.init_duals_primals_ub().copy()
        duals_slacks_lb = interface.init_duals_slacks_lb().copy()
        duals_slacks_ub = interface.init_duals_slacks_ub().copy()
        self.process_init(primals, interface.primals_lb(), interface.primals_ub())
        self.process_init(slacks, interface.ineq_lb(), interface.ineq_ub())
        self.process_init_duals_lb(duals_primals_lb, self.interface.primals_lb())
        self.process_init_duals_ub(duals_primals_ub, self.interface.primals_ub())
        self.process_init_duals_lb(duals_slacks_lb, self.interface.ineq_lb())
        self.process_init_duals_ub(duals_slacks_ub, self.interface.ineq_ub())
        interface.set_barrier_parameter(self._barrier_parameter)
        alpha_primal_max = 1
        alpha_dual_max = 1
        self.logger.info('{_iter:<6}{objective:<11}{primal_inf:<11}{dual_inf:<11}{compl_inf:<11}{barrier:<11}{alpha_p:<11}{alpha_d:<11}{reg:<11}{time:<7}'.format(_iter='Iter', objective='Objective', primal_inf='Prim Inf', dual_inf='Dual Inf', compl_inf='Comp Inf', barrier='Barrier', alpha_p='Prim Step', alpha_d='Dual Step', reg='Reg', time='Time'))
        reg_coef = 0
        timer.stop('init')
        status = InteriorPointStatus.error
        for _iter in range(max_iter):
            self._iter = _iter
            interface.set_primals(primals)
            interface.set_slacks(slacks)
            interface.set_duals_eq(duals_eq)
            interface.set_duals_ineq(duals_ineq)
            interface.set_duals_primals_lb(duals_primals_lb)
            interface.set_duals_primals_ub(duals_primals_ub)
            interface.set_duals_slacks_lb(duals_slacks_lb)
            interface.set_duals_slacks_ub(duals_slacks_ub)
            timer.start('convergence check')
            primal_inf, dual_inf, complimentarity_inf = self.check_convergence(barrier=0, timer=timer)
            timer.stop('convergence check')
            objective = interface.evaluate_objective()
            self.logger.info('{_iter:<6}{objective:<11.2e}{primal_inf:<11.2e}{dual_inf:<11.2e}{compl_inf:<11.2e}{barrier:<11.2e}{alpha_p:<11.2e}{alpha_d:<11.2e}{reg:<11.2e}{time:<7.3f}'.format(_iter=_iter, objective=objective, primal_inf=primal_inf, dual_inf=dual_inf, compl_inf=complimentarity_inf, barrier=self._barrier_parameter, alpha_p=alpha_primal_max, alpha_d=alpha_dual_max, reg=reg_coef, time=time.time() - t0))
            if max(primal_inf, dual_inf, complimentarity_inf) <= tol:
                status = InteriorPointStatus.optimal
                break
            timer.start('convergence check')
            primal_inf, dual_inf, complimentarity_inf = self.check_convergence(barrier=self._barrier_parameter, timer=timer)
            timer.stop('convergence check')
            if max(primal_inf, dual_inf, complimentarity_inf) <= 0.1 * self._barrier_parameter:
                self.update_barrier_parameter()
            interface.set_barrier_parameter(self._barrier_parameter)
            timer.start('eval')
            timer.start('eval kkt')
            kkt = interface.evaluate_primal_dual_kkt_matrix(timer=timer)
            timer.stop('eval kkt')
            timer.start('eval rhs')
            rhs = interface.evaluate_primal_dual_kkt_rhs(timer=timer)
            timer.stop('eval rhs')
            timer.stop('eval')
            timer.start('factorize')
            reg_coef = self.factorize(kkt=kkt, timer=timer)
            timer.stop('factorize')
            timer.start('back solve')
            with self.linear_solve_context:
                self.logger.info('Iter: %s' % self._iter)
                delta, res = linear_solver.do_back_solve(rhs)
                if res.status != LinearSolverStatus.successful:
                    raise RuntimeError(f'Backsolve failed: {res.status}')
            timer.stop('back solve')
            interface.set_primal_dual_kkt_solution(delta)
            timer.start('frac boundary')
            alpha_primal_max, alpha_dual_max = self.fraction_to_the_boundary()
            timer.stop('frac boundary')
            delta_primals = interface.get_delta_primals()
            delta_slacks = interface.get_delta_slacks()
            delta_duals_eq = interface.get_delta_duals_eq()
            delta_duals_ineq = interface.get_delta_duals_ineq()
            delta_duals_primals_lb = interface.get_delta_duals_primals_lb()
            delta_duals_primals_ub = interface.get_delta_duals_primals_ub()
            delta_duals_slacks_lb = interface.get_delta_duals_slacks_lb()
            delta_duals_slacks_ub = interface.get_delta_duals_slacks_ub()
            primals += alpha_primal_max * delta_primals
            slacks += alpha_primal_max * delta_slacks
            duals_eq += alpha_dual_max * delta_duals_eq
            duals_ineq += alpha_dual_max * delta_duals_ineq
            duals_primals_lb += alpha_dual_max * delta_duals_primals_lb
            duals_primals_ub += alpha_dual_max * delta_duals_primals_ub
            duals_slacks_lb += alpha_dual_max * delta_duals_slacks_lb
            duals_slacks_ub += alpha_dual_max * delta_duals_slacks_ub
        timer.stop('IP solve')
        if report_timing:
            print(timer)
        return status

    def factorize(self, kkt, timer=None):
        desired_n_neg_evals = self.interface.n_eq_constraints() + self.interface.n_ineq_constraints()
        reg_iter = 0
        with self.factorization_context as fact_con:
            status, num_realloc = try_factorization_and_reallocation(kkt=kkt, linear_solver=self.linear_solver, reallocation_factor=self.reallocation_factor, max_iter=self.max_reallocation_iterations, timer=timer)
            if status not in {LinearSolverStatus.successful, LinearSolverStatus.singular}:
                raise RuntimeError('Could not factorize KKT system; linear solver status: ' + str(status))
            if status == LinearSolverStatus.successful:
                neg_eig = self.linear_solver.get_inertia()[1]
            else:
                neg_eig = None
            fact_con.log_info(_iter=self._iter, reg_iter=reg_iter, num_realloc=num_realloc, coef=0, neg_eig=neg_eig, status=status)
            reg_iter += 1
            if status == LinearSolverStatus.singular:
                constraint_reg_coef = self.base_eq_reg_coef * self._barrier_parameter ** 0.25
                kkt = self.interface.regularize_equality_gradient(kkt=kkt, coef=constraint_reg_coef, copy_kkt=False)
            total_hess_reg_coef = self.hess_reg_coef
            last_hess_reg_coef = 0
            while neg_eig != desired_n_neg_evals or status == LinearSolverStatus.singular:
                kkt = self.interface.regularize_hessian(kkt=kkt, coef=total_hess_reg_coef - last_hess_reg_coef, copy_kkt=False)
                status, num_realloc = try_factorization_and_reallocation(kkt=kkt, linear_solver=self.linear_solver, reallocation_factor=self.reallocation_factor, max_iter=self.max_reallocation_iterations, timer=timer)
                if status != LinearSolverStatus.successful:
                    raise RuntimeError('Could not factorize KKT system; linear solver status: ' + str(status))
                neg_eig = self.linear_solver.get_inertia()[1]
                fact_con.log_info(_iter=self._iter, reg_iter=reg_iter, num_realloc=num_realloc, coef=total_hess_reg_coef, neg_eig=neg_eig, status=status)
                reg_iter += 1
                if reg_iter > self.max_reg_iter:
                    raise RuntimeError('Exceeded maximum number of regularization iterations.')
                last_hess_reg_coef = total_hess_reg_coef
                total_hess_reg_coef *= self.reg_factor_increase
        return last_hess_reg_coef

    def process_init(self, x, lb, ub):
        process_init(x, lb, ub)

    def process_init_duals_lb(self, x, lb):
        process_init_duals_lb(x, lb)

    def process_init_duals_ub(self, x, ub):
        process_init_duals_ub(x, ub)

    def check_convergence(self, barrier, timer=None):
        """
        Parameters
        ----------
        barrier: float
        timer: HierarchicalTimer

        Returns
        -------
        primal_inf: float
        dual_inf: float
        complimentarity_inf: float
        """
        if timer is None:
            timer = HierarchicalTimer()
        interface = self.interface
        slacks = interface.get_slacks()
        timer.start('grad obj')
        grad_obj = interface.get_obj_factor() * interface.evaluate_grad_objective()
        timer.stop('grad obj')
        timer.start('jac eq')
        jac_eq = interface.evaluate_jacobian_eq()
        timer.stop('jac eq')
        timer.start('jac ineq')
        jac_ineq = interface.evaluate_jacobian_ineq()
        timer.stop('jac ineq')
        timer.start('eq cons')
        eq_resid = interface.evaluate_eq_constraints()
        timer.stop('eq cons')
        timer.start('ineq cons')
        ineq_resid = interface.evaluate_ineq_constraints() - slacks
        timer.stop('ineq cons')
        primals = interface.get_primals()
        duals_eq = interface.get_duals_eq()
        duals_ineq = interface.get_duals_ineq()
        duals_primals_lb = interface.get_duals_primals_lb()
        duals_primals_ub = interface.get_duals_primals_ub()
        duals_slacks_lb = interface.get_duals_slacks_lb()
        duals_slacks_ub = interface.get_duals_slacks_ub()
        primals_lb = interface.primals_lb()
        primals_ub = interface.primals_ub()
        primals_lb_mod = primals_lb.copy()
        primals_ub_mod = primals_ub.copy()
        primals_lb_mod[np.isneginf(primals_lb)] = 0
        primals_ub_mod[np.isinf(primals_ub)] = 0
        ineq_lb = interface.ineq_lb()
        ineq_ub = interface.ineq_ub()
        ineq_lb_mod = ineq_lb.copy()
        ineq_ub_mod = ineq_ub.copy()
        ineq_lb_mod[np.isneginf(ineq_lb)] = 0
        ineq_ub_mod[np.isinf(ineq_ub)] = 0
        timer.start('grad_lag_primals')
        grad_lag_primals = grad_obj + jac_eq.transpose() * duals_eq
        grad_lag_primals += jac_ineq.transpose() * duals_ineq
        grad_lag_primals -= duals_primals_lb
        grad_lag_primals += duals_primals_ub
        timer.stop('grad_lag_primals')
        timer.start('grad_lag_slacks')
        grad_lag_slacks = -duals_ineq - duals_slacks_lb + duals_slacks_ub
        timer.stop('grad_lag_slacks')
        timer.start('bound resids')
        primals_lb_resid = (primals - primals_lb_mod) * duals_primals_lb - barrier
        primals_ub_resid = (primals_ub_mod - primals) * duals_primals_ub - barrier
        primals_lb_resid[np.isneginf(primals_lb)] = 0
        primals_ub_resid[np.isinf(primals_ub)] = 0
        slacks_lb_resid = (slacks - ineq_lb_mod) * duals_slacks_lb - barrier
        slacks_ub_resid = (ineq_ub_mod - slacks) * duals_slacks_ub - barrier
        slacks_lb_resid[np.isneginf(ineq_lb)] = 0
        slacks_ub_resid[np.isinf(ineq_ub)] = 0
        timer.stop('bound resids')
        if eq_resid.size == 0:
            max_eq_resid = 0
        else:
            max_eq_resid = np.max(np.abs(eq_resid))
        if ineq_resid.size == 0:
            max_ineq_resid = 0
        else:
            max_ineq_resid = np.max(np.abs(ineq_resid))
        primal_inf = max(max_eq_resid, max_ineq_resid)
        max_grad_lag_primals = np.max(np.abs(grad_lag_primals))
        if grad_lag_slacks.size == 0:
            max_grad_lag_slacks = 0
        else:
            max_grad_lag_slacks = np.max(np.abs(grad_lag_slacks))
        dual_inf = max(max_grad_lag_primals, max_grad_lag_slacks)
        if primals_lb_resid.size == 0:
            max_primals_lb_resid = 0
        else:
            max_primals_lb_resid = np.max(np.abs(primals_lb_resid))
        if primals_ub_resid.size == 0:
            max_primals_ub_resid = 0
        else:
            max_primals_ub_resid = np.max(np.abs(primals_ub_resid))
        if slacks_lb_resid.size == 0:
            max_slacks_lb_resid = 0
        else:
            max_slacks_lb_resid = np.max(np.abs(slacks_lb_resid))
        if slacks_ub_resid.size == 0:
            max_slacks_ub_resid = 0
        else:
            max_slacks_ub_resid = np.max(np.abs(slacks_ub_resid))
        complimentarity_inf = max(max_primals_lb_resid, max_primals_ub_resid, max_slacks_lb_resid, max_slacks_ub_resid)
        return (primal_inf, dual_inf, complimentarity_inf)

    def fraction_to_the_boundary(self):
        return fraction_to_the_boundary(self.interface, 1 - self._barrier_parameter)