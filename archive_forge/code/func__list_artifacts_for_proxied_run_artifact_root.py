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
@catch_mlflow_exception
def _list_artifacts_for_proxied_run_artifact_root(proxied_artifact_root, relative_path=None):
    """
    Lists artifacts from the specified ``relative_path`` within the specified proxied Run artifact
    root (i.e. a Run artifact root with scheme ``http``, ``https``, or ``mlflow-artifacts``).

    Args:
        proxied_artifact_root: The Run artifact root location (URI) with scheme ``http``,
                               ``https``, or ``mlflow-artifacts`` that can be resolved by the
                               MLflow server to a concrete storage location.
        relative_path: The relative path within the specified ``proxied_artifact_root`` under
                       which to list artifact contents. If ``None``, artifacts are listed from
                       the ``proxied_artifact_root`` directory.
    """
    parsed_proxied_artifact_root = urllib.parse.urlparse(proxied_artifact_root)
    assert parsed_proxied_artifact_root.scheme in ['http', 'https', 'mlflow-artifacts']
    artifact_destination_repo = _get_artifact_repo_mlflow_artifacts()
    artifact_destination_path = _get_proxied_run_artifact_destination_path(proxied_artifact_root=proxied_artifact_root, relative_path=relative_path)
    artifact_entities = []
    for file_info in artifact_destination_repo.list_artifacts(artifact_destination_path):
        basename = posixpath.basename(file_info.path)
        run_relative_artifact_path = posixpath.join(relative_path, basename) if relative_path else basename
        artifact_entities.append(FileInfo(run_relative_artifact_path, file_info.is_dir, file_info.file_size))
    return artifact_entities