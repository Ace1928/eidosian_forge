import numpy as np
from ase.optimize.optimize import Dynamics
def _compute_update_with_fs(self, potentiostat_step_size):
    """Uses the Frenet–Serret formulas to perform curvature based
        extrapolation to compute the update vector"""
    delta_r = self.r - self.rold
    delta_s = np.linalg.norm(delta_r)
    delta_T = self.T - self.Told
    delta_N = self.N - self.Nold
    dTds = delta_T / delta_s
    dNds = delta_N / delta_s
    if self.use_tangent_curvature:
        curvature = np.linalg.norm(dTds)
    else:
        curvature = np.linalg.norm(dNds)
    self.curvature = curvature
    if self.angle_limit is not None:
        phi = np.pi / 180 * self.angle_limit
        self.step_size = np.sqrt(2 - 2 * np.cos(phi)) / curvature
        self.step_size = min(self.step_size, self.maxstep)
    delta_s_perpendicular, delta_s_parallel, delta_s_drift = self.compute_step_contributions(potentiostat_step_size)
    N_guess = self.N + dNds * delta_s_parallel
    T_guess = self.T + dTds * delta_s_parallel
    N_guess = normalize(N_guess)
    T_guess = normalize(T_guess)
    dr_perpendicular = delta_s_perpendicular * N_guess
    dr_parallel = delta_s_parallel * self.T * (1 - (delta_s_parallel * curvature) ** 2 / 6.0) + self.N * (curvature / 2.0) * delta_s_parallel ** 2
    D = self.create_drift_unit_vector(N_guess, T_guess)
    dr_drift = D * delta_s_drift
    dr = dr_perpendicular + dr_parallel + dr_drift
    dr = self.step_size * normalize(dr)
    return dr