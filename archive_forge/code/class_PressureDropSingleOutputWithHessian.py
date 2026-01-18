from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from ..external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock
class PressureDropSingleOutputWithHessian(PressureDropSingleOutput):

    def __init__(self):
        super(PressureDropSingleOutputWithHessian, self).__init__()
        self._output_con_mult_values = np.zeros(1, dtype=np.float64)

    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        np.copyto(self._output_con_mult_values, output_con_multiplier_values)

    def evaluate_hessian_outputs(self):
        c = self._input_values[1]
        F = self._input_values[2]
        irow = np.asarray([2, 2], dtype=np.int64)
        jcol = np.asarray([1, 2], dtype=np.int64)
        data = self._output_con_mult_values[0] * np.asarray([-8 * F, -8 * c], dtype=np.float64)
        hess = spa.coo_matrix((data, (irow, jcol)), shape=(3, 3))
        return hess