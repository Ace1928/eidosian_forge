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
def _log_run_param(self, run_info, param):
    param_path = self._get_param_path(run_info.experiment_id, run_info.run_id, param.key)
    writeable_param_value = self._writeable_value(param.value)
    if os.path.exists(param_path):
        self._validate_new_param_value(param_path=param_path, param_key=param.key, run_id=run_info.run_id, new_value=writeable_param_value)
    make_containing_dirs(param_path)
    write_to(param_path, writeable_param_value)