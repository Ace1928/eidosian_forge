from the paper `Michael Roeder, Andreas Both and Alexander Hinneburg: "Exploring the space of topic coherence measures"
import logging
import multiprocessing as mp
from collections import namedtuple
import numpy as np
from gensim import interfaces, matutils
from gensim import utils
from gensim.topic_coherence import (
from gensim.topic_coherence.probability_estimation import unique_ids_from_segments
def _get_topics(self):
    """Internal helper function to return topics from a trained topic model."""
    return self._get_topics_from_model(self.model, self.topn)