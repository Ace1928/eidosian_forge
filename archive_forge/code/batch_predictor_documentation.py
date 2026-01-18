import logging
from typing import Type
from ray.train import Checkpoint
from ray.train.predictor import Predictor
from ray.util.annotations import Deprecated
Batch predictor class.

    Takes a predictor class and a checkpoint and provides an interface to run
    batch scoring on Datasets.

    This batch predictor wraps around a predictor class and executes it
    in a distributed way when calling ``predict()``.
    