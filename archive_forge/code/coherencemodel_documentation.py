from the paper `Michael Roeder, Andreas Both and Alexander Hinneburg: "Exploring the space of topic coherence measures"
import logging
import multiprocessing as mp
from collections import namedtuple
import numpy as np
from gensim import interfaces, matutils
from gensim import utils
from gensim.topic_coherence import (
from gensim.topic_coherence.probability_estimation import unique_ids_from_segments
Get average topic and model coherences.

        Parameters
        ----------
        model_topics : list of list of str
            Topics from the model.

        Returns
        -------
        list of (float, float)
            Sequence of pairs of average topic coherence and average coherence.

        