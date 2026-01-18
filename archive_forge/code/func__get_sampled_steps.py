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
def _get_sampled_steps(run_ids, metric_key, max_results):
    start_step = args.get('start_step')
    end_step = args.get('end_step')
    if start_step is not None and end_step is not None:
        start_step = int(start_step)
        end_step = int(end_step)
        if start_step > end_step:
            raise MlflowException.invalid_parameter_value(f'end_step must be greater than start_step. Found start_step={start_step} and end_step={end_step}.')
    elif start_step is not None or end_step is not None:
        raise MlflowException.invalid_parameter_value('If either start step or end step are specified, both must be specified.')
    all_runs = [[m.step for m in store.get_metric_history(run_id, metric_key)] for run_id in run_ids]
    all_mins_and_maxes = {step for run in all_runs if run for step in [min(run), max(run)]}
    all_steps = sorted({step for sublist in all_runs for step in sublist})
    if start_step is None and end_step is None:
        start_step = 0
        end_step = all_steps[-1] if all_steps else 0
    all_mins_and_maxes = {step for step in all_mins_and_maxes if start_step <= step <= end_step}
    sampled_steps = _get_sampled_steps_from_steps(start_step, end_step, max_results, all_steps)
    return sorted(sampled_steps.union(all_mins_and_maxes))