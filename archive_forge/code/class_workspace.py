from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
class workspace(object):
    """
    OSQP solver workspace

    Attributes
    ----------
    data                   - scaled QP problem
    info                   - solver information
    linsys_solver          - structure for linear system solution
    scaling                - scaling matrices
    settings               - settings structure
    solution               - solution structure


    Additional workspace variables
    ------------------------------
    first_run              - flag to indicate if it is the first run
    clear_update_time      - flag to indicate if update_time should be cleared
    timer                  - saved time instant for timing purposes
    x                      - primal iterate
    x_prev                 - previous primal iterate
    xz_tilde               - x_tilde and z_tilde iterates stacked together
    y                      - dual iterate
    z                      - z iterate
    z_prev                 - previous z iterate

    Vectorized rho parameter
    ------------------------
    rho_vec                - vector of rho values for each constraint
    rho_inv_vec            - vector of reciprocal rho values
    constr_type            - type of constraints: loose (-1), eq (1), ineq (0)

    Primal infeasibility related workspace variables
    ------------------------------------------------
    delta_y                - difference of consecutive y
    Atdelta_y              - A' * delta_y

    Dual infeasibility related workspace variables
    ----------------------------------------------
    delta_x                - difference of consecutive x
    Pdelta_x               - P * delta_x
    Adelta_x               - A * delta_x

    """