from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from . import (evaluation_io, MultiClassEvaluation, SumEvaluation,
from .onsets import onset_evaluation, OnsetEvaluation
from ..io import load_notes
class NoteEvaluation(MultiClassEvaluation):
    """
    Evaluation class for measuring Precision, Recall and F-measure of notes.

    Parameters
    ----------
    detections : str, list or numpy array
        Detected notes.
    annotations : str, list or numpy array
        Annotated ground truth notes.
    window : float, optional
        F-measure evaluation window [seconds]
    delay : float, optional
        Delay the detections `delay` seconds for evaluation.

    """

    def __init__(self, detections, annotations, window=WINDOW, delay=0, **kwargs):
        detections = np.array(detections, dtype=np.float, ndmin=2)
        annotations = np.array(annotations, dtype=np.float, ndmin=2)
        if delay != 0:
            detections[:, 0] += delay
        numbers = note_onset_evaluation(detections, annotations, window)
        tp, fp, tn, fn, errors = numbers
        super(NoteEvaluation, self).__init__(tp, fp, tn, fn, **kwargs)
        self.errors = errors
        self.detections = detections
        self.annotations = annotations
        self.window = window

    @property
    def mean_error(self):
        """Mean of the errors."""
        warnings.warn('mean_error is given for all notes, this will change!')
        if len(self.errors) == 0:
            return np.nan
        return np.mean(self.errors[:, 0])

    @property
    def std_error(self):
        """Standard deviation of the errors."""
        warnings.warn('std_error is given for all notes, this will change!')
        if len(self.errors) == 0:
            return np.nan
        return np.std(self.errors[:, 0])

    def tostring(self, notes=False, **kwargs):
        """

        Parameters
        ----------
        notes : bool, optional
            Display detailed output for all individual notes.

        Returns
        -------
        str
            Evaluation metrics formatted as a human readable string.

        """
        ret = ''
        if self.name is not None:
            ret += '%s\n  ' % self.name
        if notes:
            notes = []
            if self.tp.any():
                notes = np.append(notes, np.unique(self.tp[:, 1]))
            if self.fp.any():
                notes = np.append(notes, np.unique(self.fp[:, 1]))
            if self.tn.any():
                notes = np.append(notes, np.unique(self.tn[:, 1]))
            if self.fn.any():
                notes = np.append(notes, np.unique(self.fn[:, 1]))
            for note in sorted(np.unique(notes)):
                det = self.detections[self.detections[:, 1] == note][:, 0]
                ann = self.annotations[self.annotations[:, 1] == note][:, 0]
                name = 'MIDI note %s' % note
                e = OnsetEvaluation(det, ann, self.window, name=name)
                ret += '  %s\n' % e.tostring(notes=False)
        ret += 'Notes: %5d TP: %5d FP: %4d FN: %4d Precision: %.3f Recall: %.3f F-measure: %.3f Acc: %.3f mean: %5.1f ms std: %5.1f ms' % (self.num_annotations, self.num_tp, self.num_fp, self.num_fn, self.precision, self.recall, self.fmeasure, self.accuracy, self.mean_error * 1000.0, self.std_error * 1000.0)
        return ret