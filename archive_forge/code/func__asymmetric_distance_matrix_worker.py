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
def _asymmetric_distance_matrix_worker(worker_id, entire_ttda, ttdas_sent, n_ttdas, masking_method, masking_threshold, pipe):
    """Worker that computes the distance to all other nodes from a chunk of nodes."""
    logger.info(f'spawned worker {worker_id} to generate {n_ttdas} rows of the asymmetric distance matrix')
    ttda1 = entire_ttda[ttdas_sent:ttdas_sent + n_ttdas]
    distance_chunk = _calculate_asymmetric_distance_matrix_chunk(ttda1=ttda1, ttda2=entire_ttda, start_index=ttdas_sent, masking_method=masking_method, masking_threshold=masking_threshold)
    pipe.send((worker_id, distance_chunk))
    pipe.close()