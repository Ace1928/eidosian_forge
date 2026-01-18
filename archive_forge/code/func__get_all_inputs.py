import json
import logging
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional
from mlflow.entities import (
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities.run_info import check_run_is_active
from mlflow.environment_variables import MLFLOW_TRACKING_DIR
from mlflow.exceptions import MissingConfigException, MlflowException
from mlflow.protos import databricks_pb2
from mlflow.protos.databricks_pb2 import (
from mlflow.protos.internal_pb2 import InputVertexType
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry.file_store import FileStore as ModelRegistryFileStore
from mlflow.store.tracking import (
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.utils import get_results_from_paginated_fn, insecure_hash
from mlflow.utils.file_utils import (
from mlflow.utils.mlflow_tags import (
from mlflow.utils.name_utils import _generate_random_name, _generate_unique_integer_id
from mlflow.utils.search_utils import SearchExperimentsUtils, SearchUtils
from mlflow.utils.string_utils import is_string_type
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import (
from mlflow.utils.validation import (
def _get_all_inputs(self, run_info: RunInfo) -> RunInputs:
    run_dir = self._get_run_dir(run_info.experiment_id, run_info.run_id)
    inputs_parent_path = os.path.join(run_dir, FileStore.INPUTS_FOLDER_NAME)
    experiment_dir = self._get_experiment_path(run_info.experiment_id, assert_exists=True)
    datasets_parent_path = os.path.join(experiment_dir, FileStore.DATASETS_FOLDER_NAME)
    if not os.path.exists(inputs_parent_path) or not os.path.exists(datasets_parent_path):
        return RunInputs(dataset_inputs=[])
    dataset_dirs = os.listdir(datasets_parent_path)
    dataset_inputs = []
    for input_dir in os.listdir(inputs_parent_path):
        input_dir_full_path = os.path.join(inputs_parent_path, input_dir)
        fs_input = FileStore._FileStoreInput.from_yaml(input_dir_full_path, FileStore.META_DATA_FILE_NAME)
        if fs_input.source_type != InputVertexType.DATASET:
            logging.warning(f"Encountered invalid run input source type '{fs_input.source_type}'. Skipping.")
            continue
        matching_dataset_dirs = [d for d in dataset_dirs if d == fs_input.source_id]
        if not matching_dataset_dirs:
            logging.warning(f"Failed to find dataset with ID '{fs_input.source_id}' referenced as an input of the run with ID '{run_info.run_id}'. Skipping.")
            continue
        elif len(matching_dataset_dirs) > 1:
            logging.warning(f"Found multiple datasets with ID '{fs_input.source_id}'. Using the first one.")
        dataset_dir = matching_dataset_dirs[0]
        dataset = FileStore._get_dataset_from_dir(datasets_parent_path, dataset_dir)
        dataset_input = DatasetInput(dataset=dataset, tags=[InputTag(key=key, value=value) for key, value in fs_input.tags.items()])
        dataset_inputs.append(dataset_input)
    return RunInputs(dataset_inputs=dataset_inputs)