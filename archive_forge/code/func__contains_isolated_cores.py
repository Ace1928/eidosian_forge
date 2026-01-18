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
def _contains_isolated_cores(label, cluster, min_cores):
    """Check if the cluster has at least ``min_cores`` of cores that belong to no other cluster."""
    return sum([neighboring_labels == {label} for neighboring_labels in cluster.neighboring_labels]) >= min_cores