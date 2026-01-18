from __future__ import absolute_import, division, print_function
import numpy as np
from . import Evaluation, MeanEvaluation, SumEvaluation, evaluation_io
from ..io import load_onsets
from ..utils import combine_events
class OnsetMeanEvaluation(MeanEvaluation, OnsetSumEvaluation):
    """
    Class for averaging onset evaluations.

    """

    @property
    def mean_error(self):
        """Mean of the errors."""
        return np.nanmean([e.mean_error for e in self.eval_objects])

    @property
    def std_error(self):
        """Standard deviation of the errors."""
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
        ret += 'Onsets: %5.2f TP: %5.2f FP: %5.2f FN: %5.2f Precision: %.3f Recall: %.3f F-measure: %.3f mean: %5.1f ms std: %5.1f ms' % (self.num_annotations, self.num_tp, self.num_fp, self.num_fn, self.precision, self.recall, self.fmeasure, self.mean_error * 1000.0, self.std_error * 1000.0)
        return ret