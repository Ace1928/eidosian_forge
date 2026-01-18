from __future__ import absolute_import, division, print_function
from functools import wraps
import warnings
import numpy as np
from . import (MeanEvaluation, calc_absolute_errors, calc_errors,
from .onsets import OnsetEvaluation
from ..io import load_beats
class BeatMeanEvaluation(MeanEvaluation):
    """
    Class for averaging beat evaluation scores.

    """
    METRIC_NAMES = BeatEvaluation.METRIC_NAMES

    @property
    def fmeasure(self):
        """F-measure."""
        return np.nanmean([e.fmeasure for e in self.eval_objects])

    @property
    def pscore(self):
        """P-score."""
        return np.nanmean([e.pscore for e in self.eval_objects])

    @property
    def cemgil(self):
        """Cemgil accuracy."""
        return np.nanmean([e.cemgil for e in self.eval_objects])

    @property
    def goto(self):
        """Goto accuracy."""
        return np.nanmean([e.goto for e in self.eval_objects])

    @property
    def cmlc(self):
        """CMLc."""
        return np.nanmean([e.cmlc for e in self.eval_objects])

    @property
    def cmlt(self):
        """CMLt."""
        return np.nanmean([e.cmlt for e in self.eval_objects])

    @property
    def amlc(self):
        """AMLc."""
        return np.nanmean([e.amlc for e in self.eval_objects])

    @property
    def amlt(self):
        """AMLt."""
        return np.nanmean([e.amlt for e in self.eval_objects])

    @property
    def information_gain(self):
        """Information gain."""
        return np.nanmean([e.information_gain for e in self.eval_objects])

    @property
    def error_histogram(self):
        """Error histogram."""
        if not self.eval_objects:
            return np.zeros(0)
        return np.sum([e.error_histogram for e in self.eval_objects], axis=0)

    @property
    def global_information_gain(self):
        """Global information gain."""
        if len(self.error_histogram) == 0:
            return 0.0
        return _information_gain(self.error_histogram)

    def tostring(self, **kwargs):
        return tostring(self)