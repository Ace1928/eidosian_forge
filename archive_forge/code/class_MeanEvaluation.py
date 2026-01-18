from __future__ import absolute_import, division, print_function
import numpy as np
from . import chords, beats, notes, onsets, tempo
from .beats import BeatEvaluation, BeatMeanEvaluation
from .chords import ChordEvaluation, ChordMeanEvaluation, ChordSumEvaluation
from .key import KeyEvaluation, KeyMeanEvaluation
from .notes import NoteEvaluation, NoteMeanEvaluation, NoteSumEvaluation
from .onsets import OnsetEvaluation, OnsetMeanEvaluation, OnsetSumEvaluation
from .tempo import TempoEvaluation, TempoMeanEvaluation
class MeanEvaluation(SumEvaluation):
    """
    Simple class for averaging evaluation.

    Parameters
    ----------
    eval_objects : list
        Evaluation objects.
    name : str
        Name to be displayed.

    """

    def __init__(self, eval_objects, name=None, **kwargs):
        super(MeanEvaluation, self).__init__(eval_objects, **kwargs)
        self.name = name or 'mean for %d files' % len(self)

    @property
    def num_tp(self):
        """Number of true positive detections."""
        if not self.eval_objects:
            return 0.0
        return np.nanmean([e.num_tp for e in self.eval_objects])

    @property
    def num_fp(self):
        """Number of false positive detections."""
        if not self.eval_objects:
            return 0.0
        return np.nanmean([e.num_fp for e in self.eval_objects])

    @property
    def num_tn(self):
        """Number of true negative detections."""
        if not self.eval_objects:
            return 0.0
        return np.nanmean([e.num_tn for e in self.eval_objects])

    @property
    def num_fn(self):
        """Number of false negative detections."""
        if not self.eval_objects:
            return 0.0
        return np.nanmean([e.num_fn for e in self.eval_objects])

    @property
    def num_annotations(self):
        """Number of annotations."""
        if not self.eval_objects:
            return 0.0
        return np.nanmean([e.num_annotations for e in self.eval_objects])

    @property
    def precision(self):
        """Precision."""
        return np.nanmean([e.precision for e in self.eval_objects])

    @property
    def recall(self):
        """Recall."""
        return np.nanmean([e.recall for e in self.eval_objects])

    @property
    def fmeasure(self):
        """F-measure."""
        return np.nanmean([e.fmeasure for e in self.eval_objects])

    @property
    def accuracy(self):
        """Accuracy."""
        return np.nanmean([e.accuracy for e in self.eval_objects])

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
        ret = 'Annotations: %5.2f TP: %5.2f FP: %5.2f FN: %5.2fPrecision: %.3f Recall: %.3f F-measure: %.3f Acc: %.3f' % (self.num_annotations, self.num_tp, self.num_fp, self.num_fn, self.precision, self.recall, self.fmeasure, self.accuracy)
        return ret