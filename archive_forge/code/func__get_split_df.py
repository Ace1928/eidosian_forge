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
def _get_split_df(input_df, hash_buckets, split_ratios):
    train_ratio, validation_ratio, test_ratio = split_ratios
    ratio_sum = train_ratio + validation_ratio + test_ratio
    train_bucket_end = train_ratio / ratio_sum
    validation_bucket_end = (train_ratio + validation_ratio) / ratio_sum
    train_df = input_df[hash_buckets.map(lambda x: x < train_bucket_end)]
    validation_df = input_df[hash_buckets.map(lambda x: train_bucket_end <= x < validation_bucket_end)]
    test_df = input_df[hash_buckets.map(lambda x: x >= validation_bucket_end)]
    empty_splits = [split_name for split_name, split_df in [('train split', train_df), ('validation split', validation_df), ('test split', test_df)] if len(split_df) == 0]
    if len(empty_splits) > 0:
        _logger.warning(f'The following input dataset splits are empty: {','.join(empty_splits)}.')
    return (train_df, validation_df, test_df)