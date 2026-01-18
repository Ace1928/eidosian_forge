from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from . import EvaluationMixin, MeanEvaluation, evaluation_io
from ..io import load_tempo
class TempoMeanEvaluation(MeanEvaluation):
    """
    Class for averaging tempo evaluation scores.

    """
    METRIC_NAMES = TempoEvaluation.METRIC_NAMES

    @property
    def pscore(self):
        """P-Score."""
        return np.nanmean([e.pscore for e in self.eval_objects])

    @property
    def any(self):
        """At least one tempo correct."""
        return np.nanmean([e.any for e in self.eval_objects])

    @property
    def all(self):
        """All tempi correct."""
        return np.nanmean([e.all for e in self.eval_objects])

    @property
    def acc1(self):
        """Accuracy 1."""
        return np.nanmean([e.acc1 for e in self.eval_objects])

    @property
    def acc2(self):
        """Accuracy 2."""
        return np.nanmean([e.acc2 for e in self.eval_objects])

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