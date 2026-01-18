import base64
import functools
import logging
import os
import shutil
from contextlib import contextmanager
import mlflow
from mlflow.entities import Run
from mlflow.environment_variables import MLFLOW_UNITY_CATALOG_PRESIGNED_URLS_ENABLED
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_service_pb2 import UcModelRegistryService
from mlflow.protos.service_pb2 import GetRun, MlflowService
from mlflow.store.artifact.presigned_url_artifact_repo import PresignedUrlArtifactRepository
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry.rest_store import BaseRestStore
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils._unity_catalog_utils import (
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_host_creds, is_databricks_uri
from mlflow.utils.mlflow_tags import (
from mlflow.utils.proto_json_utils import message_to_json, parse_dict
from mlflow.utils.rest_utils import (
def _validate_model_signature(self, local_model_path):
    model = _load_model(local_model_path)
    signature_required_explanation = 'All models in the Unity Catalog must be logged with a model signature containing both input and output type specifications. See https://mlflow.org/docs/latest/models.html#model-signature for details on how to log a model with a signature'
    if model.signature is None:
        raise MlflowException(f'Model passed for registration did not contain any signature metadata. {signature_required_explanation}')
    if model.signature.outputs is None:
        raise MlflowException(f'Model passed for registration contained a signature that includes only inputs. {signature_required_explanation}')