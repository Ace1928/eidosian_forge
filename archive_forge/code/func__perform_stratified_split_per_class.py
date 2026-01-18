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
def _perform_stratified_split_per_class(input_df, split_ratios, target_col):
    classes = np.unique(input_df[target_col])
    partial_func = partial(_perform_split_for_one_class, input_df=input_df, split_ratios=split_ratios, target_col=target_col)
    with ThreadPool(os.cpu_count() or _NUM_DEFAULT_CPUS) as p:
        zipped_dfs = p.map(partial_func, classes)
        train_df, validation_df, test_df = [pd.concat(x) for x in list(zip(*zipped_dfs))]
        return (train_df, validation_df, test_df)