import numpy as np
from scipy.optimize._trustregion_exact import (
from scipy.linalg import (svd, get_lapack_funcs, det, qr, norm)
from numpy.testing import (assert_array_equal,
class TestIterativeSubproblem:

    def test_for_the_easy_case(self):
        H = [[10, 2, 3, 4], [2, 1, 7, 1], [3, 7, 1, 7], [4, 1, 7, 2]]
        g = [1, 1, 1, 1]
        trust_radius = 1
        subprob = IterativeSubproblem(x=0, fun=lambda x: 0, jac=lambda x: np.array(g), hess=lambda x: np.array(H), k_easy=1e-10, k_hard=1e-10)
        p, hits_boundary = subprob.solve(trust_radius)
        assert_array_almost_equal(p, [0.00393332, -0.55260862, 0.67065477, -0.49480341])
        assert_array_almost_equal(hits_boundary, True)

    def test_for_the_hard_case(self):
        H = [[10, 2, 3, 4], [2, 1, 7, 1], [3, 7, 1, 7], [4, 1, 7, 2]]
        g = [6.485264152132744, 1, 1, 1]
        s = -8.215151987441661
        trust_radius = 1
        subprob = IterativeSubproblem(x=0, fun=lambda x: 0, jac=lambda x: np.array(g), hess=lambda x: np.array(H), k_easy=1e-10, k_hard=1e-10)
        p, hits_boundary = subprob.solve(trust_radius)
        assert_array_almost_equal(-s, subprob.lambda_current)

    def test_for_interior_convergence(self):
        H = [[1.812159, 0.82687265, 0.21838879, -0.52487006, 0.25436988], [0.82687265, 2.66380283, 0.31508988, -0.40144163, 0.08811588], [0.21838879, 0.31508988, 2.38020726, -0.3166346, 0.27363867], [-0.52487006, -0.40144163, -0.3166346, 1.61927182, -0.42140166], [0.25436988, 0.08811588, 0.27363867, -0.42140166, 1.33243101]]
        g = [0.75798952, 0.01421945, 0.33847612, 0.83725004, -0.47909534]
        subprob = IterativeSubproblem(x=0, fun=lambda x: 0, jac=lambda x: np.array(g), hess=lambda x: np.array(H))
        p, hits_boundary = subprob.solve(1.1)
        assert_array_almost_equal(p, [-0.68585435, 0.1222621, -0.22090999, -0.67005053, 0.31586769])
        assert_array_almost_equal(hits_boundary, False)
        assert_array_almost_equal(subprob.lambda_current, 0)
        assert_array_almost_equal(subprob.niter, 1)

    def test_for_jac_equal_zero(self):
        H = [[0.88547534, 2.90692271, 0.98440885, -0.78911503, -0.28035809], [2.90692271, -0.04618819, 0.32867263, -0.83737945, 0.17116396], [0.98440885, 0.32867263, -0.87355957, -0.06521957, -1.43030957], [-0.78911503, -0.83737945, -0.06521957, -1.645709, -0.33887298], [-0.28035809, 0.17116396, -1.43030957, -0.33887298, -1.68586978]]
        g = [0, 0, 0, 0, 0]
        subprob = IterativeSubproblem(x=0, fun=lambda x: 0, jac=lambda x: np.array(g), hess=lambda x: np.array(H), k_easy=1e-10, k_hard=1e-10)
        p, hits_boundary = subprob.solve(1.1)
        assert_array_almost_equal(p, [0.06910534, -0.01432721, -0.65311947, -0.23815972, -0.84954934])
        assert_array_almost_equal(hits_boundary, True)

    def test_for_jac_very_close_to_zero(self):
        H = [[0.88547534, 2.90692271, 0.98440885, -0.78911503, -0.28035809], [2.90692271, -0.04618819, 0.32867263, -0.83737945, 0.17116396], [0.98440885, 0.32867263, -0.87355957, -0.06521957, -1.43030957], [-0.78911503, -0.83737945, -0.06521957, -1.645709, -0.33887298], [-0.28035809, 0.17116396, -1.43030957, -0.33887298, -1.68586978]]
        g = [0, 0, 0, 0, 1e-15]
        subprob = IterativeSubproblem(x=0, fun=lambda x: 0, jac=lambda x: np.array(g), hess=lambda x: np.array(H), k_easy=1e-10, k_hard=1e-10)
        p, hits_boundary = subprob.solve(1.1)
        assert_array_almost_equal(p, [0.06910534, -0.01432721, -0.65311947, -0.23815972, -0.84954934])
        assert_array_almost_equal(hits_boundary, True)

    def test_for_random_entries(self):
        np.random.seed(1)
        n = 5
        for case in ('easy', 'hard', 'jac_equal_zero'):
            eig_limits = [(-20, -15), (-10, -5), (-10, 0), (-5, 5), (-10, 10), (0, 10), (5, 10), (15, 20)]
            for min_eig, max_eig in eig_limits:
                H, g = random_entry(n, min_eig, max_eig, case)
                trust_radius_list = [0.1, 0.3, 0.6, 0.8, 1, 1.2, 3.3, 5.5, 10]
                for trust_radius in trust_radius_list:
                    subprob_ac = IterativeSubproblem(0, lambda x: 0, lambda x: g, lambda x: H, k_easy=1e-10, k_hard=1e-10)
                    p_ac, hits_boundary_ac = subprob_ac.solve(trust_radius)
                    J_ac = 1 / 2 * np.dot(p_ac, np.dot(H, p_ac)) + np.dot(g, p_ac)
                    stop_criteria = [(0.1, 2), (0.5, 1.1), (0.9, 1.01)]
                    for k_opt, k_trf in stop_criteria:
                        k_easy = min(k_trf - 1, 1 - np.sqrt(k_opt))
                        k_hard = 1 - k_opt
                        subprob = IterativeSubproblem(0, lambda x: 0, lambda x: g, lambda x: H, k_easy=k_easy, k_hard=k_hard)
                        p, hits_boundary = subprob.solve(trust_radius)
                        J = 1 / 2 * np.dot(p, np.dot(H, p)) + np.dot(g, p)
                        if hits_boundary:
                            assert_array_equal(np.abs(norm(p) - trust_radius) <= (k_trf - 1) * trust_radius, True)
                        else:
                            assert_equal(norm(p) <= trust_radius, True)
                        assert_equal(J <= k_opt * J_ac, True)