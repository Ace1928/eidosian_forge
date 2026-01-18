from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from . import EvaluationMixin, MeanEvaluation, evaluation_io
from ..io import load_tempo
@property
def acc2(self):
    """Accuracy 2."""
    return np.nanmean([e.acc2 for e in self.eval_objects])