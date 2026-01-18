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
def _create_experiment_with_id(self, name, experiment_id, artifact_uri, tags):
    if not artifact_uri:
        resolved_artifact_uri = append_to_uri_path(self.artifact_root_uri, str(experiment_id))
    else:
        resolved_artifact_uri = resolve_uri_if_local(artifact_uri)
    meta_dir = mkdir(self.root_directory, str(experiment_id))
    creation_time = get_current_time_millis()
    experiment = Experiment(experiment_id, name, resolved_artifact_uri, LifecycleStage.ACTIVE, creation_time=creation_time, last_update_time=creation_time)
    experiment_dict = dict(experiment)
    del experiment_dict['tags']
    write_yaml(meta_dir, FileStore.META_DATA_FILE_NAME, experiment_dict)
    if tags is not None:
        for tag in tags:
            self.set_experiment_tag(experiment_id, tag)
    return experiment_id