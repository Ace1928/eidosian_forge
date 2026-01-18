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
def _aggregate_topics(grouped_by_labels):
    """Aggregate the labeled topics to a list of clusters.

    Parameters
    ----------
    grouped_by_labels : dict of (int, list of :class:`Topic`)
        The return value of _group_by_labels. A mapping of the label to a list of each topic which belongs to the
        label.

    Returns
    -------
    list of :class:`Cluster`
        It is sorted by max_num_neighboring_labels in descending order. There is one single element for each cluster.

    """
    clusters = []
    for label, topics in grouped_by_labels.items():
        max_num_neighboring_labels = 0
        neighboring_labels = []
        for topic in topics:
            max_num_neighboring_labels = max(topic.num_neighboring_labels, max_num_neighboring_labels)
            neighboring_labels.append(topic.neighboring_labels)
        neighboring_labels = [x for x in neighboring_labels if len(x) > 0]
        clusters.append(Cluster(max_num_neighboring_labels=max_num_neighboring_labels, neighboring_labels=neighboring_labels, label=label, num_cores=len([topic for topic in topics if topic.is_core])))
    logger.info('found %s clusters', len(clusters))
    return clusters