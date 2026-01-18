import logging
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_FP_config
from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm
from pyomo.core import ConstraintList
from pyomo.contrib.mindtpy.util import calc_jacobians
from pyomo.opt import SolverFactory
from pyomo.contrib.mindtpy.cut_generation import add_oa_cuts
def add_cuts(self, dual_values, linearize_active=True, linearize_violated=True, cb_opt=None, nlp=None):
    add_oa_cuts(self.mip, dual_values, self.jacobians, self.objective_sense, self.mip_constraint_polynomial_degree, self.mip_iter, self.config, self.timing, cb_opt, linearize_active, linearize_violated)