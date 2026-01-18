from __future__ import absolute_import, division, print_function
import numpy as np
from . import chords, beats, notes, onsets, tempo
from .beats import BeatEvaluation, BeatMeanEvaluation
from .chords import ChordEvaluation, ChordMeanEvaluation, ChordSumEvaluation
from .key import KeyEvaluation, KeyMeanEvaluation
from .notes import NoteEvaluation, NoteMeanEvaluation, NoteSumEvaluation
from .onsets import OnsetEvaluation, OnsetMeanEvaluation, OnsetSumEvaluation
from .tempo import TempoEvaluation, TempoMeanEvaluation
class SumEvaluation(SimpleEvaluation):
    """
    Simple class for summing evaluations.

    Parameters
    ----------
    eval_objects : list
        Evaluation objects.
    name : str
        Name to be displayed.

    """

    def __init__(self, eval_objects, name=None):
        if not isinstance(eval_objects, list):
            eval_objects = [eval_objects]
        self.eval_objects = eval_objects
        self.name = name or 'sum for %d files' % len(self)

    def __len__(self):
        return len(self.eval_objects)

    @property
    def num_tp(self):
        """Number of true positive detections."""
        return sum((e.num_tp for e in self.eval_objects))

    @property
    def num_fp(self):
        """Number of false positive detections."""
        return sum((e.num_fp for e in self.eval_objects))

    @property
    def num_tn(self):
        """Number of true negative detections."""
        return sum((e.num_tn for e in self.eval_objects))

    @property
    def num_fn(self):
        """Number of false negative detections."""
        return sum((e.num_fn for e in self.eval_objects))

    @property
    def num_annotations(self):
        """Number of annotations."""
        return sum((e.num_annotations for e in self.eval_objects))