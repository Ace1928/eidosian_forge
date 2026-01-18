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
class CBDBSCAN:
    """A Variation of the DBSCAN algorithm called Checkback DBSCAN (CBDBSCAN).

    The algorithm works based on DBSCAN-like parameters 'eps' and 'min_samples' that respectively define how far a
    "nearby" point is, and the minimum number of nearby points needed to label a candidate datapoint a core of a
    cluster. (See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).

    The algorithm works as follows:

    1. (A)symmetric distance matrix provided at fit-time (called 'amatrix').
       For the sake of example below, assume the there are only five topics (amatrix contains distances with dim 5x5),
       T_1, T_2, T_3, T_4, T_5:
    2. Start by scanning a candidate topic with respect to a parent topic
       (e.g. T_1 with respect to parent None)
    3. Check which topics are nearby the candidate topic using 'self.eps' as a threshold and call them neighbours
       (e.g. assume T_3, T_4, and T_5 are nearby and become neighbours)
    4. If there are more neighbours than 'self.min_samples', the candidate topic becomes a core candidate for a cluster
       (e.g. if 'min_samples'=1, then T_1 becomes the first core of a cluster)
    5. If candidate is a core, CheckBack (CB) to find the fraction of neighbours that are either the parent or the
       parent's neighbours.  If this fraction is more than 75%, give the candidate the same label as its parent.
       (e.g. in the trivial case there is no parent (or neighbours of that parent), a new incremental label is given)
    6. If candidate is a core, recursively scan the next nearby topic (e.g. scan T_3) labeling the previous topic as
       the parent and the previous neighbours as the parent_neighbours - repeat steps 2-6:

       2. (e.g. Scan candidate T_3 with respect to parent T_1 that has parent_neighbours T_3, T_4, and T_5)
       3. (e.g. T5 is the only neighbour)
       4. (e.g. number of neighbours is 1, therefore candidate T_3 becomes a core)
       5. (e.g. CheckBack finds that two of the four parent and parent neighbours are neighbours of candidate T_3.
          Therefore the candidate T_3 does NOT get the same label as its parent T_1)
       6. (e.g. Scan candidate T_5 with respect to parent T_3 that has parent_neighbours T_5)

    The CB step has the effect that it enforces cluster compactness and allows the model to avoid creating clusters for
    unstable topics made of a composition of multiple stable topics.

    """

    def __init__(self, eps, min_samples):
        """Create a new CBDBSCAN object. Call fit in order to train it on an asymmetric distance matrix.

        Parameters
        ----------
        eps : float
            epsilon for the CBDBSCAN algorithm, having the same meaning as in classic DBSCAN clustering.
        min_samples : int
            The minimum number of samples in the neighborhood of a topic to be considered a core in CBDBSCAN.

        """
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, amatrix):
        """Apply the algorithm to an asymmetric distance matrix."""
        self.next_label = 0
        topic_clustering_results = [Topic(is_core=False, neighboring_labels=set(), neighboring_topic_indices=set(), label=None, num_neighboring_labels=0, valid_neighboring_labels=set()) for i in range(len(amatrix))]
        amatrix_copy = amatrix.copy()
        np.fill_diagonal(amatrix_copy, 1)
        min_distance_per_topic = [(distance, index) for index, distance in enumerate(amatrix_copy.min(axis=1))]
        min_distance_per_topic_sorted = sorted(min_distance_per_topic, key=lambda distance: distance[0])
        ordered_min_similarity = [index for distance, index in min_distance_per_topic_sorted]

        def scan_topic(topic_index, current_label=None, parent_neighbors=None):
            """Extend the cluster in one direction.

            Results are accumulated to ``self.results``.

            Parameters
            ----------
            topic_index : int
                The topic that might be added to the existing cluster, or which might create a new cluster if necessary.
            current_label : int
                The label of the cluster that might be suitable for ``topic_index``

            """
            neighbors_sorted = sorted([(distance, index) for index, distance in enumerate(amatrix_copy[topic_index])], key=lambda x: x[0])
            neighboring_topic_indices = [index for distance, index in neighbors_sorted if distance < self.eps]
            num_neighboring_topics = len(neighboring_topic_indices)
            if num_neighboring_topics >= self.min_samples:
                topic_clustering_results[topic_index].is_core = True
                if current_label is None:
                    current_label = self.next_label
                    self.next_label += 1
                else:
                    close_parent_neighbors_mask = amatrix_copy[topic_index][parent_neighbors] < self.eps
                    if close_parent_neighbors_mask.mean() < 0.25:
                        current_label = self.next_label
                        self.next_label += 1
                topic_clustering_results[topic_index].label = current_label
                for neighboring_topic_index in neighboring_topic_indices:
                    if topic_clustering_results[neighboring_topic_index].label is None:
                        ordered_min_similarity.remove(neighboring_topic_index)
                        scan_topic(neighboring_topic_index, current_label, neighboring_topic_indices + [topic_index])
                    topic_clustering_results[neighboring_topic_index].neighboring_topic_indices.add(topic_index)
                    topic_clustering_results[neighboring_topic_index].neighboring_labels.add(current_label)
            elif current_label is None:
                topic_clustering_results[topic_index].label = -1
            else:
                topic_clustering_results[topic_index].label = current_label
        while len(ordered_min_similarity) != 0:
            next_topic_index = ordered_min_similarity.pop(0)
            scan_topic(next_topic_index)
        self.results = topic_clustering_results