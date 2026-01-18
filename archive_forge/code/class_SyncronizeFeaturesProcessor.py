from __future__ import absolute_import, division, print_function
import sys
import warnings
import numpy as np
from .beats_hmm import (BarStateSpace, BarTransitionModel,
from ..ml.hmm import HiddenMarkovModel
from ..processors import ParallelProcessor, Processor, SequentialProcessor
from ..utils import string_types
class SyncronizeFeaturesProcessor(Processor):
    """
    Synchronize features to beats.

    First, divide a beat interval into `beat_subdivision` divisions. Then
    summarise all features that fall into one subdivision. If no feature value
    for a subdivision is found, it is set to 0.

    Parameters
    ----------
    beat_subdivisions : int
        Number of subdivisions a beat is divided into.
    fps : float
        Frames per second.

    """

    def __init__(self, beat_subdivisions, fps, **kwargs):
        self.beat_subdivisions = beat_subdivisions
        self.fps = fps

    def process(self, data, **kwargs):
        """
        Synchronize features to beats.

        Average all feature values that fall into a window of beat duration /
        beat subdivisions, centered on the beat positions or interpolated
        subdivisions, starting with the first beat.

        Parameters
        ----------
        data : tuple (features, beats)
            Tuple of two numpy arrays, the first containing features to be
            synchronized and second the beat times.

        Returns
        -------
        numpy array (num beats - 1, beat subdivisions, features dim.)
            Beat synchronous features.

        """
        features, beats = data
        if beats.size == 0:
            return (np.array([]), np.array([]))
        if beats.ndim > 1:
            beats = beats[:, 0]
        while float(len(features)) / self.fps < beats[-1]:
            beats = beats[:-1]
            warnings.warn('Beat sequence too long compared to features.')
        num_beats = len(beats)
        features = np.array(features.T, copy=False, ndmin=2).T
        feat_dim = features.shape[-1]
        beat_features = np.zeros((num_beats - 1, self.beat_subdivisions, feat_dim))
        beat_start = int(max(0, np.floor((beats[0] - 0.02) * self.fps)))
        for i in range(num_beats - 1):
            beat_duration = beats[i + 1] - beats[i]
            offset = 0.5 * beat_duration / self.beat_subdivisions
            offset = np.min([offset, 0.05])
            beat_end = int(np.floor((beats[i + 1] - offset) * self.fps))
            subdiv = np.floor(np.linspace(0, self.beat_subdivisions, beat_end - beat_start, endpoint=False))
            beat = features[beat_start:beat_end]
            subdiv_features = [beat[subdiv == div] for div in range(self.beat_subdivisions)]
            beat_features[i, :, :] = np.array([np.mean(x, axis=0) for x in subdiv_features])
            beat_start = beat_end
        return beat_features