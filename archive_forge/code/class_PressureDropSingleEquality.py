from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from ..external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock
class PressureDropSingleEquality(ExternalGreyBoxModel):

    def __init__(self):
        self._input_names = ['Pin', 'c', 'F', 'Pout']
        self._input_values = np.zeros(4, dtype=np.float64)
        self._equality_constraint_names = ['pdrop']

    def input_names(self):
        return self._input_names

    def equality_constraint_names(self):
        return self._equality_constraint_names

    def set_input_values(self, input_values):
        assert len(input_values) == 4
        np.copyto(self._input_values, input_values)

    def evaluate_equality_constraints(self):
        Pin = self._input_values[0]
        c = self._input_values[1]
        F = self._input_values[2]
        Pout = self._input_values[3]
        return np.asarray([Pout - (Pin - 4 * c * F ** 2)], dtype=np.float64)

    def evaluate_jacobian_equality_constraints(self):
        c = self._input_values[1]
        F = self._input_values[2]
        irow = np.asarray([0, 0, 0, 0], dtype=np.int64)
        jcol = np.asarray([0, 1, 2, 3], dtype=np.int64)
        nonzeros = np.asarray([-1, 4 * F ** 2, 4 * 2 * c * F, 1], dtype=np.float64)
        jac = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(1, 4))
        return jac