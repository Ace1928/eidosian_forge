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
def _validate_and_execute_custom_split(self, split_fn, input_df):
    custom_split_mapping_series = split_fn(input_df)
    if not isinstance(custom_split_mapping_series, pd.Series):
        raise MlflowException('Return type of the custom split function should be a pandas series', error_code=INVALID_PARAMETER_VALUE)
    copy_df = input_df.copy()
    copy_df['split'] = custom_split_mapping_series
    train_df = input_df[copy_df['split'] == SplitValues.TRAINING.value].reset_index(drop=True)
    validation_df = input_df[copy_df['split'] == SplitValues.VALIDATION.value].reset_index(drop=True)
    test_df = input_df[copy_df['split'] == SplitValues.TEST.value].reset_index(drop=True)
    if train_df.size + validation_df.size + test_df.size != input_df.size:
        incorrect_args = custom_split_mapping_series[~custom_split_mapping_series.isin([SplitValues.TRAINING.value, SplitValues.VALIDATION.value, SplitValues.TEST.value])].unique()
        raise MlflowException(f'Returned pandas series from custom split step should only contain {SplitValues.TRAINING.value}, {SplitValues.VALIDATION.value} or {SplitValues.TEST.value} as values. Value returned back: {incorrect_args}', error_code=INVALID_PARAMETER_VALUE)
    return (train_df, validation_df, test_df)