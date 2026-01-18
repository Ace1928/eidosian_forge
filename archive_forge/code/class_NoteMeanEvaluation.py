from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from . import (evaluation_io, MultiClassEvaluation, SumEvaluation,
from .onsets import onset_evaluation, OnsetEvaluation
from ..io import load_notes
class NoteMeanEvaluation(MeanEvaluation, NoteSumEvaluation):
    """
    Class for averaging note evaluations.

    """

    @property
    def mean_error(self):
        """Mean of the errors."""
        warnings.warn('mean_error is given for all notes, this will change!')
        return np.nanmean([e.mean_error for e in self.eval_objects])

    @property
    def std_error(self):
        """Standard deviation of the errors."""
        warnings.warn('std_error is given for all notes, this will change!')
        return np.nanmean([e.std_error for e in self.eval_objects])

    def tostring(self, **kwargs):
        """
        Format the evaluation metrics as a human readable string.

        Returns
        -------
        str
            Evaluation metrics formatted as a human readable string.

        """
        ret = ''
        if self.name is not None:
            ret += '%s\n  ' % self.name
        ret += 'Notes: %5.2f TP: %5.2f FP: %5.2f FN: %5.2f Precision: %.3f Recall: %.3f F-measure: %.3f Acc: %.3f mean: %5.1f ms std: %5.1f ms' % (self.num_annotations, self.num_tp, self.num_fp, self.num_fn, self.precision, self.recall, self.fmeasure, self.accuracy, self.mean_error * 1000.0, self.std_error * 1000.0)
        return ret