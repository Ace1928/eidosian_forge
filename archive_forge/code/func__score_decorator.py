from __future__ import absolute_import, division, print_function
from functools import wraps
import warnings
import numpy as np
from . import (MeanEvaluation, calc_absolute_errors, calc_errors,
from .onsets import OnsetEvaluation
from ..io import load_beats
def _score_decorator(perfect_score, zero_score):
    """
    Decorate metric with evaluation results for perfect and zero score.

    Parameters
    ----------
    perfect_score : float or tuple
    zero_score : float or tuple

    Returns
    -------
    metric
        Decorated metric.

    """

    def wrap(metric):
        """Metric to decorate"""

        @wraps(metric)
        def score(detections, annotations, *args, **kwargs):
            """
            Return perfect/zero score if neither/either detections and
            annotations are given, repsectively.

            """
            if len(detections) == 0 and len(annotations) == 0:
                return perfect_score
            elif (len(detections) == 0) != (len(annotations) == 0):
                return zero_score
            return metric(detections, annotations, *args, **kwargs)
        return score
    return wrap