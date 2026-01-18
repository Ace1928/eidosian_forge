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
def _create_hash_buckets(input_df, n_jobs=-1):
    with Timer() as t:
        hash_buckets = _hash_pandas_dataframe(input_df, n_jobs=n_jobs).map(lambda x: x % _SPLIT_HASH_BUCKET_NUM / _SPLIT_HASH_BUCKET_NUM)
    _logger.debug(f'Creating hash buckets on input dataset containing {len(input_df)} rows consumes {t:.3f} seconds.')
    return hash_buckets