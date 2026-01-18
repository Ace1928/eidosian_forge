from pyomo.common.dependencies import numpy as np
import pyomo.common.dependencies.scipy.sparse as scipy_sparse
from pyomo.common.dependencies import attempt_import
class GreyBoxModel(egb.ExternalGreyBoxModel):
    """Greybox model to compute the example objective function."""

    def __init__(self, initial, use_exact_derivatives=True, verbose=False):
        """
            Parameters

            use_exact_derivatives: bool
                If True, the exact derivatives are used. If False, the finite difference
                approximation is used.
            verbose: bool
                If True, print information about the model.
            """
        self._use_exact_derivatives = use_exact_derivatives
        self.verbose = verbose
        self.initial = initial
        self._output_con_mult_values = np.zeros(1)
        if not use_exact_derivatives:
            raise NotImplementedError('use_exact_derivatives == False not supported')

    def input_names(self):
        """Return the names of the inputs."""
        self.input_name_list = ['X1', 'X2', 'Y1', 'Y2', 'Y3']
        return self.input_name_list

    def equality_constraint_names(self):
        """Return the names of the equality constraints."""
        return []

    def output_names(self):
        """Return the names of the outputs."""
        return ['z']

    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        """Set the values of the output constraint multipliers."""
        assert len(output_con_multiplier_values) == 1
        np.copyto(self._output_con_mult_values, output_con_multiplier_values)

    def finalize_block_construction(self, pyomo_block):
        """Finalize the construction of the ExternalGreyBoxBlock."""
        if self.initial is not None:
            print('initialized')
            pyomo_block.inputs['X1'].value = self.initial['X1']
            pyomo_block.inputs['X2'].value = self.initial['X2']
            pyomo_block.inputs['Y1'].value = self.initial['Y1']
            pyomo_block.inputs['Y2'].value = self.initial['Y2']
            pyomo_block.inputs['Y3'].value = self.initial['Y3']
        else:
            print('uninitialized')
            for n in self.input_name_list:
                pyomo_block.inputs[n].value = 1
        pyomo_block.inputs['X1'].setub(4)
        pyomo_block.inputs['X1'].setlb(0)
        pyomo_block.inputs['X2'].setub(4)
        pyomo_block.inputs['X2'].setlb(0)
        pyomo_block.inputs['Y1'].setub(1)
        pyomo_block.inputs['Y1'].setlb(0)
        pyomo_block.inputs['Y2'].setub(1)
        pyomo_block.inputs['Y2'].setlb(0)
        pyomo_block.inputs['Y3'].setub(1)
        pyomo_block.inputs['Y3'].setlb(0)

    def set_input_values(self, input_values):
        """Set the values of the inputs."""
        self._input_values = list(input_values)

    def evaluate_equality_constraints(self):
        """Evaluate the equality constraints."""
        return None

    def evaluate_outputs(self):
        """Evaluate the output of the model."""
        x1 = self._input_values[0]
        x2 = self._input_values[1]
        y1 = self._input_values[2]
        y2 = self._input_values[3]
        y3 = self._input_values[4]
        z = x1 ** 2 + x2 ** 2 + y1 + 1.5 * y2 + 0.5 * y3
        if self.verbose:
            print('\n Consider inputs [x1,x2,y1,y2,y3] =\n', x1, x2, y1, y2, y3)
            print('   z = ', z, '\n')
        return np.asarray([z], dtype=np.float64)

    def evaluate_jacobian_equality_constraints(self):
        """Evaluate the Jacobian of the equality constraints."""
        return None
    '\n        def _extract_and_assemble_fim(self):\n            M = np.zeros((self.n_parameters, self.n_parameters))\n            for i in range(self.n_parameters):\n                for k in range(self.n_parameters):                \n                    M[i,k] = self._input_values[self.ele_to_order[(i,k)]]\n\n            return M\n        '

    def evaluate_jacobian_outputs(self):
        """Evaluate the Jacobian of the outputs."""
        if self._use_exact_derivatives:
            row = np.zeros(5)
            col = np.zeros(5)
            data = np.zeros(5)
            row[0], col[0], data[0] = (0, 0, 2 * self._input_values[0])
            row[0], col[1], data[1] = (0, 1, 2 * self._input_values[1])
            row[0], col[2], data[2] = (0, 2, 1)
            row[0], col[3], data[3] = (0, 3, 1.5)
            row[0], col[4], data[4] = (0, 4, 0.5)
            return scipy_sparse.coo_matrix((data, (row, col)), shape=(1, 5))