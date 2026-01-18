import importlib
import logging
import os
import sys
import time
from enum import Enum
from functools import partial
from multiprocessing.pool import Pool, ThreadPool
import numpy as np
import pandas as pd
from mlflow.exceptions import BAD_REQUEST, INVALID_PARAMETER_VALUE, MlflowException
from mlflow.recipes.artifacts import DataframeArtifact
from mlflow.recipes.cards import BaseCard
from mlflow.recipes.step import BaseStep, StepClass
from mlflow.recipes.utils.execution import get_step_output_path
from mlflow.recipes.utils.step import get_pandas_data_profiles, validate_classification_config
from mlflow.store.artifact.artifact_repo import _NUM_DEFAULT_CPUS
from mlflow.utils.time import Timer
def _make_elem_hashable(elem):
    if isinstance(elem, list):
        return tuple((_make_elem_hashable(e) for e in elem))
    elif isinstance(elem, dict):
        return tuple(((_make_elem_hashable(k), _make_elem_hashable(v)) for k, v in elem.items()))
    elif isinstance(elem, np.ndarray):
        return (elem.shape, tuple(elem.flatten(order='C')))
    else:
        return elem