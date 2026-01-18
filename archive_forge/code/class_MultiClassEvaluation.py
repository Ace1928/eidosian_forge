from __future__ import absolute_import, division, print_function
import numpy as np
from . import chords, beats, notes, onsets, tempo
from .beats import BeatEvaluation, BeatMeanEvaluation
from .chords import ChordEvaluation, ChordMeanEvaluation, ChordSumEvaluation
from .key import KeyEvaluation, KeyMeanEvaluation
from .notes import NoteEvaluation, NoteMeanEvaluation, NoteSumEvaluation
from .onsets import OnsetEvaluation, OnsetMeanEvaluation, OnsetSumEvaluation
from .tempo import TempoEvaluation, TempoMeanEvaluation
class MultiClassEvaluation(Evaluation):
    """
    Evaluation class for measuring Precision, Recall and F-measure based on
    2D numpy arrays with true/false positive/negative detections.

    Parameters
    ----------
    tp : list of tuples or numpy array, shape (num_tp, 2)
        True positive detections.
    fp : list of tuples or numpy array, shape (num_fp, 2)
        False positive detections.
    tn : list of tuples or numpy array, shape (num_tn, 2)
        True negative detections.
    fn : list of tuples or numpy array, shape (num_fn, 2)
        False negative detections.
    name : str
        Name to be displayed.

    Notes
    -----
    The second item of the tuples or the second column of the arrays denote
    the class the detection belongs to.

    """

    def __init__(self, tp=None, fp=None, tn=None, fn=None, **kwargs):
        if tp is None:
            tp = np.zeros((0, 2))
        if fp is None:
            fp = np.zeros((0, 2))
        if tn is None:
            tn = np.zeros((0, 2))
        if fn is None:
            fn = np.zeros((0, 2))
        super(MultiClassEvaluation, self).__init__(**kwargs)
        self.tp = np.asarray(tp, dtype=np.float)
        self.fp = np.asarray(fp, dtype=np.float)
        self.tn = np.asarray(tn, dtype=np.float)
        self.fn = np.asarray(fn, dtype=np.float)

    def tostring(self, verbose=False, **kwargs):
        """
        Format the evaluation metrics as a human readable string.

        Parameters
        ----------
        verbose : bool
            Add evaluation for individual classes.

        Returns
        -------
        str
            Evaluation metrics formatted as a human readable string.

        """
        ret = ''
        if verbose:
            classes = []
            if self.tp.any():
                classes = np.append(classes, np.unique(self.tp[:, 1]))
            if self.fp.any():
                classes = np.append(classes, np.unique(self.fp[:, 1]))
            if self.tn.any():
                classes = np.append(classes, np.unique(self.tn[:, 1]))
            if self.fn.any():
                classes = np.append(classes, np.unique(self.fn[:, 1]))
            for cls in sorted(np.unique(classes)):
                tp = self.tp[self.tp[:, 1] == cls]
                fp = self.fp[self.fp[:, 1] == cls]
                tn = self.tn[self.tn[:, 1] == cls]
                fn = self.fn[self.fn[:, 1] == cls]
                e = Evaluation(tp, fp, tn, fn, name='Class %s' % cls)
                ret += '  %s\n' % e.tostring(verbose=False)
        ret += 'Annotations: %5d TP: %5d FP: %4d FN: %4d Precision: %.3f Recall: %.3f F-measure: %.3f Acc: %.3f' % (self.num_annotations, self.num_tp, self.num_fp, self.num_fn, self.precision, self.recall, self.fmeasure, self.accuracy)
        return ret