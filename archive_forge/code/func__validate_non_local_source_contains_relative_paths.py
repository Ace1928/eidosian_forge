import bisect
import json
import logging
import os
import pathlib
import posixpath
import re
import tempfile
import time
import urllib
from functools import wraps
from typing import List, Set
import requests
from flask import Response, current_app, jsonify, request, send_file
from google.protobuf import descriptor
from google.protobuf.json_format import ParseError
from mlflow.entities import DatasetInput, ExperimentTag, FileInfo, Metric, Param, RunTag, ViewType
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.entities.multipart_upload import MultipartUploadPart
from mlflow.environment_variables import MLFLOW_DEPLOYMENTS_TARGET
from mlflow.exceptions import MlflowException, _UnsupportedMultipartUploadException
from mlflow.models import Model
from mlflow.protos import databricks_pb2
from mlflow.protos.databricks_pb2 import (
from mlflow.protos.mlflow_artifacts_pb2 import (
from mlflow.protos.mlflow_artifacts_pb2 import (
from mlflow.protos.model_registry_pb2 import (
from mlflow.protos.service_pb2 import (
from mlflow.server.validation import _validate_content_type
from mlflow.store.artifact.artifact_repo import MultipartUploadMixin
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.tracking._model_registry import utils as registry_utils
from mlflow.tracking._model_registry.registry import ModelRegistryStoreRegistry
from mlflow.tracking._tracking_service import utils
from mlflow.tracking._tracking_service.registry import TrackingStoreRegistry
from mlflow.tracking.registry import UnsupportedModelRegistryStoreURIException
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow.utils.mime_type_utils import _guess_mime_type
from mlflow.utils.promptlab_utils import _create_promptlab_run_impl
from mlflow.utils.proto_json_utils import message_to_json, parse_dict
from mlflow.utils.string_utils import is_string_type
from mlflow.utils.uri import is_local_uri, validate_path_is_safe, validate_query_string
from mlflow.utils.validation import _validate_batch_log_api_req
def _validate_non_local_source_contains_relative_paths(source: str):
    """
    Validation check to ensure that sources that are provided that conform to the schemes:
    http, https, or mlflow-artifacts do not contain relative path designations that are intended
    to access local file system paths on the tracking server.

    Example paths that this validation function is intended to find and raise an Exception if
    passed:
    "mlflow-artifacts://host:port/../../../../"
    "http://host:port/api/2.0/mlflow-artifacts/artifacts/../../../../"
    "https://host:port/api/2.0/mlflow-artifacts/artifacts/../../../../"
    "/models/artifacts/../../../"
    "s3:/my_bucket/models/path/../../other/path"
    "file://path/to/../../../../some/where/you/should/not/be"
    "mlflow-artifacts://host:port/..%2f..%2f..%2f..%2f"
    "http://host:port/api/2.0/mlflow-artifacts/artifacts%00"
    """
    invalid_source_error_message = f"Invalid model version source: '{source}'. If supplying a source as an http, https, local file path, ftp, objectstore, or mlflow-artifacts uri, an absolute path must be provided without relative path references present. Please provide an absolute path."
    while (unquoted := urllib.parse.unquote_plus(source)) != source:
        source = unquoted
    source_path = re.sub('/+', '/', urllib.parse.urlparse(source).path.rstrip('/'))
    if '\x00' in source_path:
        raise MlflowException(invalid_source_error_message, INVALID_PARAMETER_VALUE)
    resolved_source = pathlib.Path(source_path).resolve().as_posix()
    _, resolved_path = os.path.splitdrive(resolved_source)
    if resolved_path != source_path:
        raise MlflowException(invalid_source_error_message, INVALID_PARAMETER_VALUE)