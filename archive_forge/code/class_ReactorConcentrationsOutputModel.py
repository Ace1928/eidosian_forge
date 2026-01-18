import numpy as np
from scipy.optimize import fsolve
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel
class ReactorConcentrationsOutputModel(ExternalGreyBoxModel):

    def input_names(self):
        return ['sv', 'caf', 'k1', 'k2', 'k3']

    def output_names(self):
        return ['ca', 'cb', 'cc', 'cd']

    def set_input_values(self, input_values):
        self._input_values = list(input_values)

    def finalize_block_construction(self, pyomo_block):
        pyomo_block.inputs['sv'].setlb(0)
        pyomo_block.outputs['ca'].setlb(0)
        pyomo_block.outputs['cb'].setlb(0)
        pyomo_block.outputs['cc'].setlb(0)
        pyomo_block.outputs['cd'].setlb(0)
        pyomo_block.inputs['sv'].value = 5
        pyomo_block.inputs['caf'].value = 10000
        pyomo_block.inputs['k1'].value = 5 / 6
        pyomo_block.inputs['k2'].value = 5 / 3
        pyomo_block.inputs['k3'].value = 1 / 6000
        pyomo_block.outputs['ca'].value = 1
        pyomo_block.outputs['cb'].value = 1
        pyomo_block.outputs['cc'].value = 1
        pyomo_block.outputs['cd'].value = 1

    def evaluate_outputs(self):
        sv = self._input_values[0]
        caf = self._input_values[1]
        k1 = self._input_values[2]
        k2 = self._input_values[3]
        k3 = self._input_values[4]
        ret = reactor_outlet_concentrations(sv, caf, k1, k2, k3)
        return np.asarray(ret, dtype=np.float64)

    def evaluate_jacobian_outputs(self):
        delta = 1e-06
        u0 = np.copy(self._input_values)
        y0 = self.evaluate_outputs()
        jac = np.empty((4, 5))
        u = np.copy(self._input_values)
        for j in range(len(u)):
            u[j] += delta
            self.set_input_values(u)
            yperturb = self.evaluate_outputs()
            jac_col = (yperturb - y0) / delta
            jac[:, j] = jac_col
            u[j] = u0[j]
        self.set_input_values(u0)
        row = []
        col = []
        data = []
        for r in range(4):
            for c in range(5):
                row.append(r)
                col.append(c)
                data.append(jac[r, c])
        return coo_matrix((data, (row, col)), shape=(4, 5))