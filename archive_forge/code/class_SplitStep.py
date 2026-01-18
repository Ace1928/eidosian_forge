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
class SplitStep(BaseStep):

    def _validate_and_apply_step_config(self):
        self.run_end_time = None
        self.execution_duration = None
        self.num_dropped_rows = None
        self.target_col = self.step_config.get('target_col')
        self.positive_class = self.step_config.get('positive_class')
        self.skip_data_profiling = self.step_config.get('skip_data_profiling', False)
        if self.target_col is None:
            raise MlflowException('Missing target_col config in recipe config.', error_code=INVALID_PARAMETER_VALUE)
        self.skip_data_profiling = self.step_config.get('skip_data_profiling', False)
        if 'using' in self.step_config:
            if self.step_config['using'] not in ['custom', 'split_ratios']:
                raise MlflowException(f"Invalid split step configuration value {self.step_config['using']} for key 'using'. Supported values are: ['custom', 'split_ratios']", error_code=INVALID_PARAMETER_VALUE)
        else:
            self.step_config['using'] = 'split_ratios'
        if self.step_config['using'] == 'split_ratios':
            self.split_ratios = self.step_config.get('split_ratios', [0.75, 0.125, 0.125])
            if not (isinstance(self.split_ratios, list) and len(self.split_ratios) == 3 and all((isinstance(x, (int, float)) and x > 0 for x in self.split_ratios))):
                raise MlflowException('Config split_ratios must be a list containing 3 positive numbers.')
        if 'split_method' not in self.step_config and self.step_config['using'] == 'custom':
            raise MlflowException("Missing 'split_method' configuration in the split step, which is using 'custom'.", error_code=INVALID_PARAMETER_VALUE)

    def _build_profiles_and_card(self, train_df, validation_df, test_df) -> BaseCard:
        from sklearn.utils import compute_class_weight

        def _set_target_col_as_first(df, target_col):
            columns = list(df.columns)
            col = columns.pop(columns.index(target_col))
            return df[[col] + columns]
        card = BaseCard(self.recipe_name, self.name)
        if not self.skip_data_profiling:
            train_df = _set_target_col_as_first(train_df, self.target_col)
            validation_df = _set_target_col_as_first(validation_df, self.target_col)
            test_df = _set_target_col_as_first(test_df, self.target_col)
            data_profile = get_pandas_data_profiles([['Train', train_df.reset_index(drop=True)], ['Validation', validation_df.reset_index(drop=True)], ['Test', test_df.reset_index(drop=True)]])
            card.add_tab('Compare Splits', '{{PROFILE}}').add_pandas_profile('PROFILE', data_profile)
            if self.task == 'classification':
                if self.positive_class is not None:
                    mask = train_df[self.target_col] == self.positive_class
                    dfs_for_profiles = [('Positive', train_df[mask]), ('Negative', train_df[~mask])]
                    sub_title = 'Positive vs Negative'
                else:
                    classes = np.unique(train_df[self.target_col])
                    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_df[self.target_col])
                    class_weights = list(zip(classes, class_weights))
                    class_weights = sorted(class_weights, key=lambda x: x[1], reverse=True)
                    if len(class_weights) > _MAX_CLASSES_TO_PROFILE:
                        class_weights = class_weights[:_MAX_CLASSES_TO_PROFILE]
                    dfs_for_profiles = [(name, train_df[train_df[self.target_col] == name]) for name, _ in class_weights]
                    sub_title = f'Top {min(5, len(class_weights))} Classes'
                profiles = [[str(p[0]), p[1].drop(columns=[self.target_col]).reset_index(drop=True)] for p in dfs_for_profiles]
                generated_profile = get_pandas_data_profiles(profiles)
                card.add_tab(f'Compare Training Data ({sub_title})', '{{PROFILE}}').add_pandas_profile('PROFILE', generated_profile)
        card.add_tab('Run Summary', '\n                {{ SCHEMA_LOCATION }}\n                {{ TRAIN_SPLIT_NUM_ROWS }}\n                {{ VALIDATION_SPLIT_NUM_ROWS }}\n                {{ TEST_SPLIT_NUM_ROWS }}\n                {{ NUM_DROPPED_ROWS }}\n                {{ EXE_DURATION}}\n                {{ LAST_UPDATE_TIME }}\n                ').add_markdown('NUM_DROPPED_ROWS', f'**Number of dropped rows:** `{self.num_dropped_rows}`').add_markdown('TRAIN_SPLIT_NUM_ROWS', f'**Number of train dataset rows:** `{len(train_df)}`').add_markdown('VALIDATION_SPLIT_NUM_ROWS', f'**Number of validation dataset rows:** `{len(validation_df)}`').add_markdown('TEST_SPLIT_NUM_ROWS', f'**Number of test dataset rows:** `{len(test_df)}`')
        return card

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

    def _run_custom_split(self, input_df):
        split_fn = getattr(importlib.import_module(_USER_DEFINED_SPLIT_STEP_MODULE), self.step_config['split_method'])
        return self._validate_and_execute_custom_split(split_fn, input_df)

    def _run(self, output_directory):
        run_start_time = time.time()
        ingested_data_path = get_step_output_path(recipe_root_path=self.recipe_root, step_name='ingest', relative_path=_INPUT_FILE_NAME)
        input_df = pd.read_parquet(ingested_data_path)
        validate_classification_config(self.task, self.positive_class, input_df, self.target_col)
        raw_input_num_rows = len(input_df)
        if self.target_col not in input_df.columns:
            raise MlflowException(f"Target column '{self.target_col}' not found in ingested dataset.", error_code=INVALID_PARAMETER_VALUE)
        input_df = input_df.dropna(how='any', subset=[self.target_col])
        self.num_dropped_rows = raw_input_num_rows - len(input_df)
        if self.step_config['using'] == 'custom':
            train_df, validation_df, test_df = self._run_custom_split(input_df)
        else:
            train_df, validation_df, test_df = _run_split(self.task, input_df, self.split_ratios, self.target_col)
        post_split_config = self.step_config.get('post_split_method', None)
        post_split_filter_config = self.step_config.get('post_split_filter_method', None)
        if post_split_config is not None:
            sys.path.append(self.recipe_root)
            post_split = getattr(importlib.import_module(_USER_DEFINED_SPLIT_STEP_MODULE), post_split_config)
            _logger.debug(f'Running {post_split_config} on train, validation and test datasets.')
            train_df, validation_df, test_df = _validate_user_code_output(post_split, train_df, validation_df, test_df)
        elif post_split_filter_config is not None:
            sys.path.append(self.recipe_root)
            post_split_filter = getattr(importlib.import_module(_USER_DEFINED_SPLIT_STEP_MODULE), post_split_filter_config)
            _logger.debug(f'Running {post_split_filter_config} on train, validation and test datasets.')
            train_df = train_df[post_split_filter(train_df)]
        if min(len(train_df), len(validation_df), len(test_df)) < 4:
            raise MlflowException(f'Train, validation, and testing datasets cannot be less than 4 rows. Train has {len(train_df)} rows, validation has {len(validation_df)} rows, and test has {len(test_df)} rows.', error_code=BAD_REQUEST)
        train_df.to_parquet(os.path.join(output_directory, _OUTPUT_TRAIN_FILE_NAME))
        validation_df.to_parquet(os.path.join(output_directory, _OUTPUT_VALIDATION_FILE_NAME))
        test_df.to_parquet(os.path.join(output_directory, _OUTPUT_TEST_FILE_NAME))
        self.run_end_time = time.time()
        self.execution_duration = self.run_end_time - run_start_time
        return self._build_profiles_and_card(train_df, validation_df, test_df)

    @classmethod
    def from_recipe_config(cls, recipe_config, recipe_root):
        step_config = {}
        if recipe_config.get('steps', {}).get('split', {}) is not None:
            step_config.update(recipe_config.get('steps', {}).get('split', {}))
        step_config['target_col'] = recipe_config.get('target_col')
        step_config['positive_class'] = recipe_config.get('positive_class')
        step_config['recipe'] = recipe_config.get('recipe', 'regression/v1')
        return cls(step_config, recipe_root)

    @property
    def name(self):
        return 'split'

    def get_artifacts(self):
        return [DataframeArtifact('training_data', self.recipe_root, self.name, _OUTPUT_TRAIN_FILE_NAME), DataframeArtifact('validation_data', self.recipe_root, self.name, _OUTPUT_VALIDATION_FILE_NAME), DataframeArtifact('test_data', self.recipe_root, self.name, _OUTPUT_TEST_FILE_NAME)]

    def step_class(self):
        return StepClass.TRAINING