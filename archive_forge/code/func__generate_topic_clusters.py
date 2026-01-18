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
def _generate_topic_clusters(self, eps=0.1, min_samples=None):
    """Run the CBDBSCAN algorithm on all the detected topics and label them with label-indices.

        The final approval and generation of stable topics is done in ``_generate_stable_topics()``.

        Parameters
        ----------
        eps : float
            dbscan distance scale
        min_samples : int, optional
            defaults to ``int(self.num_models / 2)``, dbscan min neighbours threshold required to consider
            a topic to be a core. Should scale with the number of models, ``self.num_models``

        """
    if min_samples is None:
        min_samples = int(self.num_models / 2)
        logger.info('fitting the clustering model, using %s for min_samples', min_samples)
    else:
        logger.info('fitting the clustering model')
    self.cluster_model = CBDBSCAN(eps=eps, min_samples=min_samples)
    self.cluster_model.fit(self.asymmetric_distance_matrix)