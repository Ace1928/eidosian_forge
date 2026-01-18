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
def _generate_stable_topics(self, min_cores=None):
    """Generate stable topics out of the clusters.

        The function finds clusters of topics using a variant of DBScan.  If a cluster has enough core topics
        (c.f. parameter ``min_cores``), then this cluster represents a stable topic.  The stable topic is specifically
        calculated as the average over all topic-term distributions of the core topics in the cluster.

        This function is the last step that has to be done in the ensemble.  After this step is complete,
        Stable topics can be retrieved afterwards using the :meth:`~gensim.models.ensemblelda.EnsembleLda.get_topics`
        method.

        Parameters
        ----------
        min_cores : int
            Minimum number of core topics needed to form a cluster that represents a stable topic.
                Using ``None`` defaults to ``min_cores = min(3, max(1, int(self.num_models /4 +1)))``

        """
    if min_cores == 0:
        min_cores = 1
    if min_cores is None:
        min_cores = min(3, max(1, int(self.num_models / 4 + 1)))
        logger.info('generating stable topics, using %s for min_cores', min_cores)
    else:
        logger.info('generating stable topics')
    cbdbscan_topics = self.cluster_model.results
    grouped_by_labels = _group_by_labels(cbdbscan_topics)
    clusters = _aggregate_topics(grouped_by_labels)
    valid_clusters = _validate_clusters(clusters, min_cores)
    valid_cluster_labels = {cluster.label for cluster in valid_clusters}
    for topic in cbdbscan_topics:
        topic.valid_neighboring_labels = {label for label in topic.neighboring_labels if label in valid_cluster_labels}
    valid_core_mask = np.vectorize(_is_valid_core)(cbdbscan_topics)
    valid_topics = self.ttda[valid_core_mask]
    topic_labels = np.array([topic.label for topic in cbdbscan_topics])[valid_core_mask]
    unique_labels = np.unique(topic_labels)
    num_stable_topics = len(unique_labels)
    stable_topics = np.empty((num_stable_topics, len(self.id2word)))
    for label_index, label in enumerate(unique_labels):
        topics_of_cluster = np.array([topic for t, topic in enumerate(valid_topics) if topic_labels[t] == label])
        stable_topics[label_index] = topics_of_cluster.mean(axis=0)
    self.valid_clusters = valid_clusters
    self.stable_topics = stable_topics
    logger.info('found %s stable topics', len(stable_topics))