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
class FileModelVersion(ModelVersion):

    def __init__(self, storage_location=None, **kwargs):
        super().__init__(**kwargs)
        self._storage_location = storage_location

    @property
    def storage_location(self):
        """String. The storage location of the model version."""
        return self._storage_location

    @storage_location.setter
    def storage_location(self, location):
        self._storage_location = location

    @classmethod
    def _properties(cls):
        return sorted(ModelVersion._properties() + cls._get_properties_helper())

    def to_mlflow_entity(self):
        meta = dict(self)
        return ModelVersion.from_dictionary({**meta, 'tags': [ModelVersionTag(k, v) for k, v in meta['tags'].items()]})