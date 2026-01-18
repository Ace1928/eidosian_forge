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
def _group_by_labels(cbdbscan_topics):
    """Group all the learned cores by their label, which was assigned in the cluster_model.

    Parameters
    ----------
    cbdbscan_topics : list of :class:`Topic`
        A list of topic data resulting from fitting a :class:`~CBDBSCAN` object.
        After calling .fit on a CBDBSCAN model, the results can be retrieved from it by accessing the .results
        member, which can be used as the argument to this function. It is a list of infos gathered during
        the clustering step and each element in the list corresponds to a single topic.

    Returns
    -------
    dict of (int, list of :class:`Topic`)
        A mapping of the label to a list of topics that belong to that particular label. Also adds
        a new member to each topic called num_neighboring_labels, which is the number of
        neighboring_labels of that topic.

    """
    grouped_by_labels = {}
    for topic in cbdbscan_topics:
        if topic.is_core:
            topic.num_neighboring_labels = len(topic.neighboring_labels)
            label = topic.label
            if label not in grouped_by_labels:
                grouped_by_labels[label] = []
            grouped_by_labels[label].append(topic)
    return grouped_by_labels