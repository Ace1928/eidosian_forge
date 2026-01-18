import numpy as np
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import (DivExpression, MulExpression,
from cvxpy.atoms.affine.sum import Sum
from cvxpy.atoms.affine.trace import trace
from cvxpy.atoms.elementwise.exp import exp
from cvxpy.atoms.elementwise.log import log
from cvxpy.atoms.elementwise.maximum import maximum
from cvxpy.atoms.elementwise.minimum import minimum
from cvxpy.atoms.elementwise.power import power
from cvxpy.atoms.elementwise.xexp import xexp
from cvxpy.atoms.eye_minus_inv import eye_minus_inv
from cvxpy.atoms.geo_mean import geo_mean
from cvxpy.atoms.gmatmul import gmatmul
from cvxpy.atoms.max import max
from cvxpy.atoms.min import min
from cvxpy.atoms.norm1 import norm1
from cvxpy.atoms.norm_inf import norm_inf
from cvxpy.atoms.one_minus_pos import one_minus_pos
from cvxpy.atoms.pf_eigenvalue import pf_eigenvalue
from cvxpy.atoms.pnorm import Pnorm
from cvxpy.atoms.prod import Prod
from cvxpy.atoms.quad_form import quad_form
from cvxpy.atoms.quad_over_lin import quad_over_lin
from cvxpy.constraints.finite_set import FiniteSet
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dgp2dcp.canonicalizers.add_canon import add_canon
from cvxpy.reductions.dgp2dcp.canonicalizers.constant_canon import (
from cvxpy.reductions.dgp2dcp.canonicalizers.div_canon import div_canon
from cvxpy.reductions.dgp2dcp.canonicalizers.exp_canon import exp_canon
from cvxpy.reductions.dgp2dcp.canonicalizers.eye_minus_inv_canon import (
from cvxpy.reductions.dgp2dcp.canonicalizers.finite_set_canon import (
from cvxpy.reductions.dgp2dcp.canonicalizers.geo_mean_canon import (
from cvxpy.reductions.dgp2dcp.canonicalizers.gmatmul_canon import gmatmul_canon
from cvxpy.reductions.dgp2dcp.canonicalizers.log_canon import log_canon
from cvxpy.reductions.dgp2dcp.canonicalizers.mul_canon import mul_canon
from cvxpy.reductions.dgp2dcp.canonicalizers.mulexpression_canon import (
from cvxpy.reductions.dgp2dcp.canonicalizers.norm1_canon import norm1_canon
from cvxpy.reductions.dgp2dcp.canonicalizers.norm_inf_canon import (
from cvxpy.reductions.dgp2dcp.canonicalizers.one_minus_pos_canon import (
from cvxpy.reductions.dgp2dcp.canonicalizers.pf_eigenvalue_canon import (
from cvxpy.reductions.dgp2dcp.canonicalizers.pnorm_canon import pnorm_canon
from cvxpy.reductions.dgp2dcp.canonicalizers.power_canon import power_canon
from cvxpy.reductions.dgp2dcp.canonicalizers.prod_canon import prod_canon
from cvxpy.reductions.dgp2dcp.canonicalizers.quad_form_canon import (
from cvxpy.reductions.dgp2dcp.canonicalizers.quad_over_lin_canon import (
from cvxpy.reductions.dgp2dcp.canonicalizers.sum_canon import sum_canon
from cvxpy.reductions.dgp2dcp.canonicalizers.trace_canon import trace_canon
from cvxpy.reductions.dgp2dcp.canonicalizers.xexp_canon import xexp_canon
from cvxpy.reductions.eliminate_pwl.canonicalizers import (
class DgpCanonMethods(dict):

    def __init__(self, *args, **kwargs) -> None:
        super(DgpCanonMethods, self).__init__(*args, **kwargs)
        self._variables = {}
        self._parameters = {}

    def __contains__(self, key):
        return key in CANON_METHODS

    def __getitem__(self, key):
        if key == Variable:
            return self.variable_canon
        elif key == Parameter:
            return self.parameter_canon
        else:
            return CANON_METHODS[key]

    def variable_canon(self, variable, args):
        del args
        if variable in self._variables:
            return (self._variables[variable], [])
        else:
            log_variable = Variable(variable.shape, var_id=variable.id)
            self._variables[variable] = log_variable
            return (log_variable, [])

    def parameter_canon(self, parameter, args):
        del args
        if parameter in self._parameters:
            return (self._parameters[parameter], [])
        else:
            log_parameter = Parameter(parameter.shape, name=parameter.name(), value=np.log(parameter.value))
            self._parameters[parameter] = log_parameter
            return (log_parameter, [])