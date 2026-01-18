from pyomo.contrib.pynumero.interfaces.nlp import NLP, ExtendedNLP
import numpy as np
import scipy.sparse as sp
class ProjectedExtendedNLP(ProjectedNLP, _ExtendedNLPDelegator):

    def __init__(self, original_nlp, primals_ordering):
        super(ProjectedExtendedNLP, self).__init__(original_nlp, primals_ordering)
        self._jacobian_eq_nz_mask = None
        self._jacobian_ineq_nz_mask = None

    def evaluate_jacobian_eq(self, out=None):
        original_jacobian = self._original_nlp.evaluate_jacobian_eq()
        if out is not None:
            np.copyto(out.data, original_jacobian.data[self._jacobian_eq_nz_mask])
            return out
        row = original_jacobian.row
        col = original_jacobian.col
        data = original_jacobian.data
        if self._jacobian_eq_nz_mask is None:
            self._jacobian_eq_nz_mask = np.isin(col, self._original_idxs)
        new_col = col[self._jacobian_eq_nz_mask]
        new_col = self._original_to_projected[new_col]
        new_row = row[self._jacobian_eq_nz_mask]
        new_data = data[self._jacobian_eq_nz_mask]
        return sp.coo_matrix((new_data, (new_row, new_col)), shape=(self.n_eq_constraints(), self.n_primals()))

    def evaluate_jacobian_ineq(self, out=None):
        original_jacobian = self._original_nlp.evaluate_jacobian_ineq()
        if out is not None:
            np.copyto(out.data, original_jacobian.data[self._jacobian_ineq_nz_mask])
            return out
        row = original_jacobian.row
        col = original_jacobian.col
        data = original_jacobian.data
        if self._jacobian_ineq_nz_mask is None:
            self._jacobian_ineq_nz_mask = np.isin(col, self._original_idxs)
        new_col = col[self._jacobian_ineq_nz_mask]
        new_col = self._original_to_projected[new_col]
        new_row = row[self._jacobian_ineq_nz_mask]
        new_data = data[self._jacobian_ineq_nz_mask]
        return sp.coo_matrix((new_data, (new_row, new_col)), shape=(self.n_ineq_constraints(), self.n_primals()))