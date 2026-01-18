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
def _remove_from_all_sets(label, clusters):
    """Remove a label from every set in "neighboring_labels" for each core in ``clusters``."""
    for cluster in clusters:
        for neighboring_labels_set in cluster.neighboring_labels:
            if label in neighboring_labels_set:
                neighboring_labels_set.remove(label)