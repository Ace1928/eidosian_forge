import numpy as np
from . import evaluation_io, EvaluationMixin
from ..io import load_chords
class ChordMeanEvaluation(ChordEvaluation):
    """
    Class for averaging chord evaluation scores, averaging piecewise (i.e.
    ignoring the lengths of the pieces). For a detailed description of the
    available metrics, refer to ChordEvaluation.

    Parameters
    ----------
    eval_objects : list
        Evaluation objects.
    name : str, optional
        Name to be displayed.

    """

    def __init__(self, eval_objects, name=None):
        self.name = name or 'piecewise mean for %d files' % len(eval_objects)
        self.eval_objects = eval_objects

    def length(self):
        """Number of evaluation objects."""
        return len(self.eval_objects)

    @property
    def root(self):
        return np.mean([e.root for e in self.eval_objects])

    @property
    def majmin(self):
        return np.mean([e.majmin for e in self.eval_objects])

    @property
    def majminbass(self):
        return np.mean([e.majminbass for e in self.eval_objects])

    @property
    def sevenths(self):
        return np.mean([e.sevenths for e in self.eval_objects])

    @property
    def seventhsbass(self):
        return np.mean([e.seventhsbass for e in self.eval_objects])

    @property
    def undersegmentation(self):
        return np.mean([e.undersegmentation for e in self.eval_objects])

    @property
    def oversegmentation(self):
        return np.mean([e.oversegmentation for e in self.eval_objects])

    @property
    def segmentation(self):
        return np.mean([e.segmentation for e in self.eval_objects])