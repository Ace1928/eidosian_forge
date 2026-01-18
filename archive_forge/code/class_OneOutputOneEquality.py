from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from ..external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock
class OneOutputOneEquality(ExternalGreyBoxModel):

    def __init__(self):
        self._input_names = ['u']
        self._equality_constraint_names = ['u2_con']
        self._output_names = ['o']
        self._u = None
        self._output_mult = None
        self._equality_mult = None

    def input_names(self):
        return self._input_names

    def equality_constraint_names(self):
        return self._equality_constraint_names

    def output_names(self):
        return self._output_names

    def finalize_block_construction(self, pyomo_block):
        pyomo_block.inputs['u'].set_value(1.0)
        pyomo_block.outputs['o'].set_value(1.0)

    def set_input_values(self, input_values):
        assert len(input_values) == 1
        self._u = input_values[0]

    def evaluate_equality_constraints(self):
        return np.asarray([self._u ** 2 - 1])

    def evaluate_outputs(self):
        return np.asarray([5 * self._u])

    def evaluate_jacobian_equality_constraints(self):
        irow = np.asarray([0], dtype=np.int64)
        jcol = np.asarray([0], dtype=np.int64)
        nonzeros = np.asarray([2 * self._u], dtype=np.float64)
        jac = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(1, 1))
        return jac

    def evaluate_jacobian_outputs(self):
        irow = np.asarray([0], dtype=np.int64)
        jcol = np.asarray([0], dtype=np.int64)
        nonzeros = np.asarray([5.0], dtype=np.float64)
        jac = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(1, 1))
        return jac

    def set_equality_constraint_multipliers(self, equality_con_multiplier_values):
        assert len(equality_con_multiplier_values) == 1
        self._equality_mult = equality_con_multiplier_values[0]

    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        assert len(output_con_multiplier_values) == 1
        self._output_mult = output_con_multiplier_values[0]

    def evaluate_hessian_equality_constraints(self):
        irow = np.asarray([0], dtype=np.int64)
        jcol = np.asarray([0], dtype=np.int64)
        data = np.asarray([self._equality_mult * 2.0], dtype=np.float64)
        hess = spa.coo_matrix((data, (irow, jcol)), shape=(1, 1))
        return hess

    def evaluate_hessian_outputs(self):
        irow = np.asarray([], dtype=np.int64)
        jcol = np.asarray([], dtype=np.int64)
        data = np.asarray([], dtype=np.float64)
        hess = spa.coo_matrix((data, (irow, jcol)), shape=(1, 1))
        return hess