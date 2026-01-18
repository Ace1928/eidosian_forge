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
def _get_proxied_run_artifact_destination_path(proxied_artifact_root, relative_path=None):
    """
    Resolves the specified proxied artifact location within a Run to a concrete storage location.

    Args:
        proxied_artifact_root: The Run artifact root location (URI) with scheme ``http``,
            ``https``, or `mlflow-artifacts` that can be resolved by the MLflow server to a
            concrete storage location.
        relative_path: The relative path of the destination within the specified
            ``proxied_artifact_root``. If ``None``, the destination is assumed to be
            the resolved ``proxied_artifact_root``.

    Returns:
        The storage location of the specified artifact.
    """
    parsed_proxied_artifact_root = urllib.parse.urlparse(proxied_artifact_root)
    assert parsed_proxied_artifact_root.scheme in ['http', 'https', 'mlflow-artifacts']
    if parsed_proxied_artifact_root.scheme == 'mlflow-artifacts':
        proxied_run_artifact_root_path = parsed_proxied_artifact_root.path.lstrip('/')
    else:
        mlflow_artifacts_http_route_anchor = '/api/2.0/mlflow-artifacts/artifacts/'
        assert mlflow_artifacts_http_route_anchor in parsed_proxied_artifact_root.path
        proxied_run_artifact_root_path = parsed_proxied_artifact_root.path.split(mlflow_artifacts_http_route_anchor)[1].lstrip('/')
    return posixpath.join(proxied_run_artifact_root_path, relative_path) if relative_path is not None else proxied_run_artifact_root_path