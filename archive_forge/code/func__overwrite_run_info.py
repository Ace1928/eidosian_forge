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
def _overwrite_run_info(self, run_info, deleted_time=None):
    run_dir = self._get_run_dir(run_info.experiment_id, run_info.run_id)
    run_info_dict = _make_persisted_run_info_dict(run_info)
    if deleted_time is not None:
        run_info_dict['deleted_time'] = deleted_time
    write_yaml(run_dir, FileStore.META_DATA_FILE_NAME, run_info_dict, overwrite=True)