from pyomo.contrib.pynumero.interfaces.nlp import NLP, ExtendedNLP
import numpy as np
import scipy.sparse as sp
class RenamedNLP(_BaseNLPDelegator):

    def __init__(self, original_nlp, primals_name_map):
        """
        This class takes an NLP that and allows one to rename the primals.
        It is a thin wrapper around the original NLP.

        Parameters
        ----------
        original_nlp : NLP-like
            The original NLP object that implements the NLP interface

        primals_name_map : dict of str --> str
            This is a dictionary that maps from the names
            in the original NLP class to the desired names
            for this instance.
        """
        super(RenamedNLP, self).__init__(original_nlp)
        self._primals_name_map = primals_name_map
        self._new_primals_names = None
        self._generate_new_names()

    def _generate_new_names(self):
        if self._new_primals_names is None:
            assert self._original_nlp.n_primals() == len(self._primals_name_map)
            self._new_primals_names = [self._primals_name_map[nm] for nm in self._original_nlp.primals_names()]

    def primals_names(self):
        return self._new_primals_names