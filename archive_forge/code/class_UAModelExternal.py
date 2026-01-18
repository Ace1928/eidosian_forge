import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
import scipy.sparse as spa
import numpy as np
import math
class UAModelExternal(ExternalGreyBoxModel):

    def __init__(self):
        super(UAModelExternal, self).__init__()
        self._input_names = ['Th_in', 'Th_out', 'Tc_in', 'Tc_out', 'UA', 'Q', 'lmtd', 'dT1', 'dT2']
        self._input_values = np.zeros(self.n_inputs(), dtype=np.float64)
        self._eq_constraint_names = ['dT1_con', 'dT2_con', 'lmtd_con', 'QUA_con', 'Qhot_con', 'Qcold_con']
        self._eq_constraint_multipliers = np.zeros(self.n_equality_constraints(), dtype=np.float64)
        self._Cp_h = 2131
        self._Cp_c = 4178
        self._Fh = 0.1
        self._Fc = 0.2

    def n_inputs(self):
        return len(self.input_names())

    def n_equality_constraints(self):
        return len(self.equality_constraint_names())

    def n_outputs(self):
        return 0

    def input_names(self):
        return self._input_names

    def equality_constraint_names(self):
        return self._eq_constraint_names

    def output_names(self):
        return []

    def finalize_block_construction(self, pyomo_block):
        pyomo_block.inputs['Th_in'].setlb(10)
        pyomo_block.inputs['Th_in'].set_value(100)
        pyomo_block.inputs['Th_out'].setlb(10)
        pyomo_block.inputs['Th_out'].set_value(50)
        pyomo_block.inputs['Tc_in'].setlb(10)
        pyomo_block.inputs['Tc_in'].set_value(30)
        pyomo_block.inputs['Tc_out'].setlb(10)
        pyomo_block.inputs['Tc_out'].set_value(50)
        pyomo_block.inputs['UA'].set_value(100)
        pyomo_block.inputs['Q'].setlb(0)
        pyomo_block.inputs['Q'].set_value(10000)
        pyomo_block.inputs['lmtd'].setlb(0)
        pyomo_block.inputs['lmtd'].set_value(20)
        pyomo_block.inputs['dT1'].setlb(0)
        pyomo_block.inputs['dT1'].set_value(20)
        pyomo_block.inputs['dT2'].setlb(0)
        pyomo_block.inputs['dT2'].set_value(20)

    def set_input_values(self, input_values):
        assert len(input_values) == 9
        np.copyto(self._input_values, input_values)

    def set_equality_constraint_multipliers(self, eq_con_multiplier_values):
        assert len(eq_con_multiplier_values) == 6
        np.copyto(self._eq_constraint_multipliers, eq_con_multiplier_values)

    def evaluate_equality_constraints(self):
        Th_in = self._input_values[0]
        Th_out = self._input_values[1]
        Tc_in = self._input_values[2]
        Tc_out = self._input_values[3]
        UA = self._input_values[4]
        Q = self._input_values[5]
        lmtd = self._input_values[6]
        dT1 = self._input_values[7]
        dT2 = self._input_values[8]
        resid = np.zeros(self.n_equality_constraints())
        resid[0] = Th_in - Tc_out - dT1
        resid[1] = Th_out - Tc_in - dT2
        resid[2] = lmtd * math.log(dT2 / dT1) - (dT2 - dT1)
        resid[3] = UA * lmtd - Q
        resid[4] = self._Fh * self._Cp_h * (Th_in - Th_out) - Q
        resid[5] = -self._Fc * self._Cp_c * (Tc_in - Tc_out) - Q
        return resid

    def evaluate_jacobian_equality_constraints(self):
        Th_in = self._input_values[0]
        Th_out = self._input_values[1]
        Tc_in = self._input_values[2]
        Tc_out = self._input_values[3]
        UA = self._input_values[4]
        Q = self._input_values[5]
        lmtd = self._input_values[6]
        dT1 = self._input_values[7]
        dT2 = self._input_values[8]
        row = np.zeros(18, dtype=np.int64)
        col = np.zeros(18, dtype=np.int64)
        data = np.zeros(18, dtype=np.float64)
        idx = 0
        row[idx], col[idx], data[idx] = (0, 0, 1.0)
        idx += 1
        row[idx], col[idx], data[idx] = (0, 3, -1.0)
        idx += 1
        row[idx], col[idx], data[idx] = (0, 7, -1.0)
        idx += 1
        row[idx], col[idx], data[idx] = (1, 1, 1.0)
        idx += 1
        row[idx], col[idx], data[idx] = (1, 2, -1.0)
        idx += 1
        row[idx], col[idx], data[idx] = (1, 8, -1.0)
        idx += 1
        row[idx], col[idx], data[idx] = (2, 6, math.log(dT2 / dT1))
        idx += 1
        row[idx], col[idx], data[idx] = (2, 7, -lmtd / dT1 + 1)
        idx += 1
        row[idx], col[idx], data[idx] = (2, 8, lmtd / dT2 - 1)
        idx += 1
        row[idx], col[idx], data[idx] = (3, 4, lmtd)
        idx += 1
        row[idx], col[idx], data[idx] = (3, 5, -1.0)
        idx += 1
        row[idx], col[idx], data[idx] = (3, 6, UA)
        idx += 1
        row[idx], col[idx], data[idx] = (4, 0, self._Fh * self._Cp_h)
        idx += 1
        row[idx], col[idx], data[idx] = (4, 1, -self._Fh * self._Cp_h)
        idx += 1
        row[idx], col[idx], data[idx] = (4, 5, -1)
        idx += 1
        row[idx], col[idx], data[idx] = (5, 2, -self._Fc * self._Cp_c)
        idx += 1
        row[idx], col[idx], data[idx] = (5, 3, self._Fc * self._Cp_c)
        idx += 1
        row[idx], col[idx], data[idx] = (5, 5, -1.0)
        idx += 1
        assert idx == 18
        return spa.coo_matrix((data, (row, col)), shape=(6, 9))

    def evaluate_hessian_equality_constraints(self):
        Th_in = self._input_values[0]
        Th_out = self._input_values[1]
        Tc_in = self._input_values[2]
        Tc_out = self._input_values[3]
        UA = self._input_values[4]
        Q = self._input_values[5]
        lmtd = self._input_values[6]
        dT1 = self._input_values[7]
        dT2 = self._input_values[8]
        row = np.zeros(5, dtype=np.int64)
        col = np.zeros(5, dtype=np.int64)
        data = np.zeros(5, dtype=np.float64)
        lam = self._eq_constraint_multipliers
        idx = 0
        row[idx], col[idx], data[idx] = (7, 6, lam[2] * -1 / dT1)
        idx += 1
        row[idx], col[idx], data[idx] = (7, 7, lam[2] * lmtd / dT1 ** 2)
        idx += 1
        row[idx], col[idx], data[idx] = (8, 6, lam[2] * 1 / dT2)
        idx += 1
        row[idx], col[idx], data[idx] = (8, 8, lam[2] * -lmtd / dT2 ** 2)
        idx += 1
        row[idx], col[idx], data[idx] = (6, 4, lam[3] * 1)
        idx += 1
        assert idx == 5
        return spa.coo_matrix((data, (row, col)), shape=(9, 9))