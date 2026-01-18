import os
import pickle
import time
import numpy as np
from ray.tune import result as tune_result
from ray.rllib.algorithms.algorithm import Algorithm, AlgorithmConfig
from ray.rllib.utils.annotations import override
class _AlgorithmImportFailed(Algorithm):
    _name = 'AlgorithmImportFailed'

    def setup(self, config):
        raise ImportError(trace)