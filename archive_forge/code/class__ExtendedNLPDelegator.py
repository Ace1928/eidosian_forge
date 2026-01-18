from pyomo.contrib.pynumero.interfaces.nlp import NLP, ExtendedNLP
import numpy as np
import scipy.sparse as sp
class _ExtendedNLPDelegator(_BaseNLPDelegator):

    def __init__(self, original_nlp):
        if not isinstance(original_nlp, ExtendedNLP):
            raise TypeError('Original NLP must be an instance of ExtendedNLP to use in an _ExtendedNLPDelegator. Got type %s' % type(original_nlp))
        super().__init__(original_nlp)

    def n_eq_constraints(self):
        return self._original_nlp.n_eq_constraints()

    def n_ineq_constraints(self):
        return self._original_nlp.n_ineq_constraints()

    def evaluate_eq_constraints(self):
        return self._original_nlp.evaluate_eq_constraints()

    def evaluate_jacobian_eq(self):
        return self._original_nlp.evaluate_jacobian_eq()

    def evaluate_ineq_constraints(self):
        return self._original_nlp.evaluate_ineq_constraints()

    def evaluate_jacobian_ineq(self):
        return self._original_nlp.evaluate_jacobian_ineq()