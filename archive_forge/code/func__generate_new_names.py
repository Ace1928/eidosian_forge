from pyomo.contrib.pynumero.interfaces.nlp import NLP, ExtendedNLP
import numpy as np
import scipy.sparse as sp
def _generate_new_names(self):
    if self._new_primals_names is None:
        assert self._original_nlp.n_primals() == len(self._primals_name_map)
        self._new_primals_names = [self._primals_name_map[nm] for nm in self._original_nlp.primals_names()]