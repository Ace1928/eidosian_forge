import logging
import sys
import itertools
import warnings
from numbers import Integral
from typing import Iterable
from numpy import (
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.corpora.dictionary import Dictionary
from gensim.utils import deprecated
@staticmethod
def _log_evaluate_word_analogies(section):
    """Calculate score by section, helper for
        :meth:`~gensim.models.keyedvectors.KeyedVectors.evaluate_word_analogies`.

        Parameters
        ----------
        section : dict of (str, (str, str, str, str))
            Section given from evaluation.

        Returns
        -------
        float
            Accuracy score if at least one prediction was made (correct or incorrect).

            Or return 0.0 if there were no predictions at all in this section.

        """
    correct, incorrect = (len(section['correct']), len(section['incorrect']))
    if correct + incorrect == 0:
        return 0.0
    score = correct / (correct + incorrect)
    logger.info('%s: %.1f%% (%i/%i)', section['section'], 100.0 * score, correct, correct + incorrect)
    return score