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
def _get_experiment_path(self, experiment_id, view_type=ViewType.ALL, assert_exists=False):
    parents = []
    if view_type == ViewType.ACTIVE_ONLY or view_type == ViewType.ALL:
        parents.append(self.root_directory)
    if view_type == ViewType.DELETED_ONLY or view_type == ViewType.ALL:
        parents.append(self.trash_folder)
    for parent in parents:
        exp_list = find(parent, experiment_id, full_path=True)
        if len(exp_list) > 0:
            return exp_list[0]
    if assert_exists:
        raise MlflowException(f'Experiment {experiment_id} does not exist.', databricks_pb2.RESOURCE_DOES_NOT_EXIST)
    return None