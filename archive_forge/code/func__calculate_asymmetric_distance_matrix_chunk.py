import logging
import os
from multiprocessing import Process, Pipe, ProcessError
import importlib
from typing import Set, Optional, List
import numpy as np
from scipy.spatial.distance import cosine
from dataclasses import dataclass
from gensim import utils
from gensim.models import ldamodel, ldamulticore, basemodel
from gensim.utils import SaveLoad
def _calculate_asymmetric_distance_matrix_chunk(ttda1, ttda2, start_index, masking_method, masking_threshold):
    """Calculate an (asymmetric) distance from each topic in ``ttda1`` to each topic in ``ttda2``.

    Parameters
    ----------
    ttda1 and ttda2: 2D arrays of floats
        Two ttda matrices that are going to be used for distance calculation. Each row in ttda corresponds to one
        topic. Each cell in the resulting matrix corresponds to the distance between a topic pair.
    start_index : int
        this function might be used in multiprocessing, so start_index has to be set as ttda1 is a chunk of the
        complete ttda in that case. start_index would be 0 if ``ttda1 == self.ttda``. When self.ttda is split into
        two pieces, each 100 ttdas long, then start_index should be be 100. default is 0
    masking_method: function

    masking_threshold: float

    Returns
    -------
    2D numpy.ndarray of floats
        Asymmetric distance matrix of size ``len(ttda1)`` by ``len(ttda2)``.

    """
    distances = np.ndarray((len(ttda1), len(ttda2)))
    if ttda1.shape[0] > 0 and ttda2.shape[0] > 0:
        avg_mask_size = 0
        for ttd1_idx, ttd1 in enumerate(ttda1):
            mask = masking_method(ttd1, masking_threshold)
            ttd1_masked = ttd1[mask]
            avg_mask_size += mask.sum()
            for ttd2_idx, ttd2 in enumerate(ttda2):
                if ttd1_idx + start_index == ttd2_idx:
                    distances[ttd1_idx][ttd2_idx] = 0
                    continue
                ttd2_masked = ttd2[mask]
                if ttd2_masked.sum() <= _COSINE_DISTANCE_CALCULATION_THRESHOLD:
                    distance = 1
                else:
                    distance = cosine(ttd1_masked, ttd2_masked)
                distances[ttd1_idx][ttd2_idx] = distance
        percent = round(100 * avg_mask_size / ttda1.shape[0] / ttda1.shape[1], 1)
        logger.info(f'the given threshold of {masking_threshold} covered on average {percent}% of tokens')
    return distances