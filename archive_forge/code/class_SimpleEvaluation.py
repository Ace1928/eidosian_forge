from __future__ import absolute_import, division, print_function
import numpy as np
from . import chords, beats, notes, onsets, tempo
from .beats import BeatEvaluation, BeatMeanEvaluation
from .chords import ChordEvaluation, ChordMeanEvaluation, ChordSumEvaluation
from .key import KeyEvaluation, KeyMeanEvaluation
from .notes import NoteEvaluation, NoteMeanEvaluation, NoteSumEvaluation
from .onsets import OnsetEvaluation, OnsetMeanEvaluation, OnsetSumEvaluation
from .tempo import TempoEvaluation, TempoMeanEvaluation
class SimpleEvaluation(EvaluationMixin):
    """
    Simple Precision, Recall, F-measure and Accuracy evaluation based on the
    numbers of true/false positive/negative detections.

    Parameters
    ----------
    num_tp : int
        Number of true positive detections.
    num_fp : int
        Number of false positive detections.
    num_tn : int
        Number of true negative detections.
    num_fn : int
        Number of false negative detections.
    name : str
        Name to be displayed.

    Notes
    -----
    This class is only suitable for a 1-class evaluation problem.

    """
    METRIC_NAMES = [('num_tp', 'No. of true positives'), ('num_fp', 'No. of false positives'), ('num_tn', 'No. of true negatives'), ('num_fn', 'No. of false negatives'), ('num_annotations', 'No. Annotations'), ('precision', 'Precision'), ('recall', 'Recall'), ('fmeasure', 'F-measure'), ('accuracy', 'Accuracy')]

    def __init__(self, num_tp=0, num_fp=0, num_tn=0, num_fn=0, name=None, **kwargs):
        self._num_tp = int(num_tp)
        self._num_fp = int(num_fp)
        self._num_tn = int(num_tn)
        self._num_fn = int(num_fn)
        self.name = name

    @property
    def num_tp(self):
        """Number of true positive detections."""
        return self._num_tp

    @property
    def num_fp(self):
        """Number of false positive detections."""
        return self._num_fp

    @property
    def num_tn(self):
        """Number of true negative detections."""
        return self._num_tn

    @property
    def num_fn(self):
        """Number of false negative detections."""
        return self._num_fn

    @property
    def num_annotations(self):
        """Number of annotations."""
        return self.num_tp + self.num_fn

    def __len__(self):
        return self.num_annotations

    @property
    def precision(self):
        """Precision."""
        retrieved = float(self.num_tp + self.num_fp)
        if retrieved == 0:
            return 1.0
        return self.num_tp / retrieved

    @property
    def recall(self):
        """Recall."""
        relevant = float(self.num_tp + self.num_fn)
        if relevant == 0:
            return 1.0
        return self.num_tp / relevant

    @property
    def fmeasure(self):
        """F-measure."""
        numerator = 2.0 * self.precision * self.recall
        if numerator == 0:
            return 0.0
        return numerator / (self.precision + self.recall)

    @property
    def accuracy(self):
        """Accuracy."""
        denominator = self.num_fp + self.num_fn + self.num_tp + self.num_tn
        if denominator == 0:
            return 1.0
        numerator = float(self.num_tp + self.num_tn)
        if numerator == 0:
            return 0.0
        return numerator / denominator

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
        ret += 'Annotations: %5d TP: %5d FP: %5d FN: %5d Precision: %.3f Recall: %.3f F-measure: %.3f Acc: %.3f' % (self.num_annotations, self.num_tp, self.num_fp, self.num_fn, self.precision, self.recall, self.fmeasure, self.accuracy)
        return ret

    def __str__(self):
        return self.tostring()