from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from . import EvaluationMixin, MeanEvaluation, evaluation_io
from ..io import load_tempo
class TempoEvaluation(EvaluationMixin):
    """
    Tempo evaluation class.

    Parameters
    ----------
    detections : str, list of tuples or numpy array
        Detected tempi (rows) and their strengths (columns).
        If a file name is given, load them from this file.
    annotations : str, list or numpy array
        Annotated ground truth tempi (rows) and their strengths (columns).
        If a file name is given, load them from this file.
    tolerance : float, optional
        Evaluation tolerance (max. allowed deviation).
    double : bool, optional
        Include double and half tempo variations.
    triple : bool, optional
        Include triple and third tempo variations.
    sort : bool, optional
        Sort the tempi by their strengths (descending order).
    max_len : bool, optional
        Evaluate at most `max_len` tempi.
    name : str, optional
        Name of the evaluation to be displayed.

    Notes
    -----
    For P-Score, the number of detected tempi will be limited to the number
    of annotations (if not further limited by `max_len`).
    For Accuracy 1 & 2 only one detected tempo is used. Depending on `sort`,
    this can be either the first or the strongest one.

    """
    METRIC_NAMES = [('pscore', 'P-score'), ('any', 'one tempo correct'), ('all', 'both tempi correct'), ('acc1', 'Accuracy 1'), ('acc2', 'Accuracy 2')]

    def __init__(self, detections, annotations, tolerance=TOLERANCE, double=DOUBLE, triple=TRIPLE, sort=True, max_len=None, name=None, **kwargs):
        detections = np.array(detections, dtype=np.float, ndmin=1)
        annotations = np.array(annotations, dtype=np.float, ndmin=1)
        if sort and detections.ndim == 2:
            detections = sort_tempo(detections)
        if sort and annotations.ndim == 2:
            annotations = sort_tempo(annotations)
        if max_len:
            detections = detections[:max_len]
            annotations = annotations[:max_len]
        self.pscore, self.any, self.all = tempo_evaluation(detections, annotations, tolerance)
        self.acc1 = tempo_evaluation(detections[:1], annotations[:1], tolerance)[1]
        try:
            tempi = annotations[:1, 0].copy()
        except IndexError:
            tempi = annotations[:1].copy()
        tempi_ = tempi.copy()
        if double:
            tempi_ = np.hstack((tempi_, tempi * 2.0, tempi / 2.0))
        if triple:
            tempi_ = np.hstack((tempi_, tempi * 3.0, tempi / 3.0))
        self.acc2 = tempo_evaluation(detections[:1], tempi_, tolerance)[1]
        self.name = name

    def __len__(self):
        return 1

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
        ret += 'pscore=%.3f (one tempo: %.3f, all tempi: %.3f) acc1=%.3f acc2=%.3f' % (self.pscore, self.any, self.all, self.acc1, self.acc2)
        return ret

    def __str__(self):
        return self.tostring()