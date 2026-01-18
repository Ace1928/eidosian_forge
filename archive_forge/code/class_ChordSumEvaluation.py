import numpy as np
from . import evaluation_io, EvaluationMixin
from ..io import load_chords
class ChordSumEvaluation(ChordEvaluation):
    """
    Class for averaging Chord evaluation scores, considering the lengths
    of the pieces. For a detailed description of the available metrics,
    refer to ChordEvaluation.

    Parameters
    ----------
    eval_objects : list
        Evaluation objects.
    name : str, optional
        Name to be displayed.

    """

    def __init__(self, eval_objects, name=None):
        self.name = name or 'weighted mean for %d files' % len(eval_objects)
        self.annotations = np.hstack([e.annotations for e in eval_objects])
        self.detections = np.hstack([e.detections for e in eval_objects])
        self.durations = np.hstack([e.durations for e in eval_objects])
        un_segs = [e.undersegmentation for e in eval_objects]
        over_segs = [e.oversegmentation for e in eval_objects]
        segs = [e.segmentation for e in eval_objects]
        lens = [e.length for e in eval_objects]
        self._underseg = np.average(un_segs, weights=lens)
        self._overseg = np.average(over_segs, weights=lens)
        self._seg = np.average(segs, weights=lens)
        self._length = sum(lens)

    def length(self):
        """Length of all evaluation objects."""
        return self._length

    @property
    def segmentation(self):
        return self._seg