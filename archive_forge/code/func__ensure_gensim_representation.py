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
def _ensure_gensim_representation(self):
    """Check if stable topics and the internal gensim representation exist. Raise an error if not."""
    if self.classic_model_representation is None:
        if len(self.stable_topics) == 0:
            raise ValueError('no stable topic was detected')
        else:
            raise ValueError('use generate_gensim_representation() first')