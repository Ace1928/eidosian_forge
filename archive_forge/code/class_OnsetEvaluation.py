from __future__ import absolute_import, division, print_function
import numpy as np
from . import Evaluation, MeanEvaluation, SumEvaluation, evaluation_io
from ..io import load_onsets
from ..utils import combine_events
class OnsetEvaluation(Evaluation):
    """
    Evaluation class for measuring Precision, Recall and F-measure of onsets.

    Parameters
    ----------
    detections : str, list or numpy array
        Detected notes.
    annotations : str, list or numpy array
        Annotated ground truth notes.
    window : float, optional
        F-measure evaluation window [seconds]
    combine : float, optional
        Combine all annotated onsets within `combine` seconds.
    delay : float, optional
        Delay the detections `delay` seconds for evaluation.

    """

    def __init__(self, detections, annotations, window=WINDOW, combine=0, delay=0, **kwargs):
        detections = np.array(detections, dtype=np.float, ndmin=1)
        annotations = np.array(annotations, dtype=np.float, ndmin=1)
        if combine > 0:
            annotations = combine_events(annotations, combine)
        if delay != 0:
            detections += delay
        tp, fp, tn, fn, errors = onset_evaluation(detections, annotations, window)
        super(OnsetEvaluation, self).__init__(tp, fp, tn, fn, **kwargs)
        self.errors = errors

    @property
    def mean_error(self):
        """Mean of the errors."""
        if len(self.errors) == 0:
            return np.nan
        return np.mean(self.errors)

    @property
    def std_error(self):
        """Standard deviation of the errors."""
        if len(self.errors) == 0:
            return np.nan
        return np.std(self.errors)

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
        ret += 'Onsets: %5d TP: %5d FP: %5d FN: %5d Precision: %.3f Recall: %.3f F-measure: %.3f mean: %5.1f ms std: %5.1f ms' % (self.num_annotations, self.num_tp, self.num_fp, self.num_fn, self.precision, self.recall, self.fmeasure, self.mean_error * 1000.0, self.std_error * 1000.0)
        return ret

    def __str__(self):
        return self.tostring()