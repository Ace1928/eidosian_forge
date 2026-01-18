import logging
import os
import shutil
import sys
import time
import urllib
from os.path import join
from typing import List
from mlflow.entities.model_registry import (
from mlflow.entities.model_registry.model_version_stages import (
from mlflow.environment_variables import MLFLOW_REGISTRY_DIR
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
from mlflow.store.artifact.utils.models import _parse_model_uri
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry import (
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.utils.file_utils import (
from mlflow.utils.search_utils import SearchModelUtils, SearchModelVersionUtils, SearchUtils
from mlflow.utils.string_utils import is_string_type
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
from mlflow.utils.validation import (
def _get_registered_model_from_path(self, model_path):
    meta = FileStore._read_yaml(model_path, FileStore.META_DATA_FILE_NAME)
    meta['tags'] = self.get_all_registered_model_tags_from_path(model_path)
    meta['aliases'] = self.get_all_registered_model_aliases_from_path(model_path)
    registered_model = RegisteredModel.from_dictionary(meta)
    registered_model.latest_versions = self.get_latest_versions(os.path.basename(model_path))
    return registered_model