from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from ..external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock
class PressureDropSingleOutput(ExternalGreyBoxModel):

    def __init__(self):
        self._input_names = ['Pin', 'c', 'F']
        self._input_values = np.zeros(3, dtype=np.float64)
        self._output_names = ['Pout']

    def input_names(self):
        return self._input_names

    def equality_constraint_names(self):
        return []

    def output_names(self):
        return self._output_names

    def set_input_values(self, input_values):
        assert len(input_values) == 3
        np.copyto(self._input_values, input_values)

    def evaluate_outputs(self):
        Pin = self._input_values[0]
        c = self._input_values[1]
        F = self._input_values[2]
        Pout = Pin - 4 * c * F ** 2
        return np.asarray([Pout], dtype=np.float64)

    def evaluate_jacobian_outputs(self):
        c = self._input_values[1]
        F = self._input_values[2]
        irow = np.asarray([0, 0, 0], dtype=np.int64)
        jcol = np.asarray([0, 1, 2], dtype=np.int64)
        nonzeros = np.asarray([1, -4 * F ** 2, -4 * c * 2 * F], dtype=np.float64)
        jac = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(1, 3))
        return jac