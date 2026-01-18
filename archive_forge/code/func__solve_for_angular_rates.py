import numpy as np
from scipy.linalg import solve_banded
from ._rotation import Rotation
def _solve_for_angular_rates(self, dt, angular_rates, rotvecs):
    angular_rate_first = angular_rates[0].copy()
    A = _angular_rate_to_rotvec_dot_matrix(rotvecs)
    A_inv = _rotvec_dot_to_angular_rate_matrix(rotvecs)
    M = _create_block_3_diagonal_matrix(2 * A_inv[1:-1] / dt[1:-1, None, None], 2 * A[1:-1] / dt[1:-1, None, None], 4 * (1 / dt[:-1] + 1 / dt[1:]))
    b0 = 6 * (rotvecs[:-1] * dt[:-1, None] ** (-2) + rotvecs[1:] * dt[1:, None] ** (-2))
    b0[0] -= 2 / dt[0] * A_inv[0].dot(angular_rate_first)
    b0[-1] -= 2 / dt[-1] * A[-1].dot(angular_rates[-1])
    for iteration in range(self.MAX_ITER):
        rotvecs_dot = _matrix_vector_product_of_stacks(A, angular_rates)
        delta_beta = _angular_acceleration_nonlinear_term(rotvecs[:-1], rotvecs_dot[:-1])
        b = b0 - delta_beta
        angular_rates_new = solve_banded((5, 5), M, b.ravel())
        angular_rates_new = angular_rates_new.reshape((-1, 3))
        delta = np.abs(angular_rates_new - angular_rates[:-1])
        angular_rates[:-1] = angular_rates_new
        if np.all(delta < self.TOL * (1 + np.abs(angular_rates_new))):
            break
    rotvecs_dot = _matrix_vector_product_of_stacks(A, angular_rates)
    angular_rates = np.vstack((angular_rate_first, angular_rates[:-1]))
    return (angular_rates, rotvecs_dot)