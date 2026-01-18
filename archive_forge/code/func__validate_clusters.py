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
def _validate_clusters(clusters, min_cores):
    """Check which clusters from the cbdbscan step are significant enough. is_valid is set accordingly."""

    def _cluster_sort_key(cluster):
        return (cluster.max_num_neighboring_labels, cluster.num_cores, cluster.label)
    sorted_clusters = sorted(clusters, key=_cluster_sort_key, reverse=False)
    for cluster in sorted_clusters:
        cluster.is_valid = None
        if cluster.num_cores < min_cores:
            cluster.is_valid = False
            _remove_from_all_sets(cluster.label, sorted_clusters)
    for cluster in [cluster for cluster in sorted_clusters if cluster.is_valid is None]:
        label = cluster.label
        if _contains_isolated_cores(label, cluster, min_cores):
            cluster.is_valid = True
        else:
            cluster.is_valid = False
            _remove_from_all_sets(label, sorted_clusters)
    return [cluster for cluster in sorted_clusters if cluster.is_valid]