import functools
import importlib
import logging
import re
import uuid
from typing import Any, Callable, Dict, Optional, Union
import sqlalchemy
from flask import Flask, Response, flash, jsonify, make_response, render_template_string, request
from werkzeug.datastructures import Authorization
from mlflow import MlflowException
from mlflow.entities import Experiment
from mlflow.entities.model_registry import RegisteredModel
from mlflow.protos.databricks_pb2 import (
from mlflow.protos.model_registry_pb2 import (
from mlflow.protos.service_pb2 import (
from mlflow.server import app
from mlflow.server.auth.config import read_auth_config
from mlflow.server.auth.logo import MLFLOW_LOGO
from mlflow.server.auth.permissions import MANAGE, Permission, get_permission
from mlflow.server.auth.routes import (
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore
from mlflow.server.handlers import (
from mlflow.store.entities import PagedList
from mlflow.utils.proto_json_utils import message_to_json, parse_dict
from mlflow.utils.rest_utils import _REST_API_PATH_PREFIX
from mlflow.utils.search_utils import SearchUtils
@catch_mlflow_exception
def _before_request():
    if is_unprotected_route(request.path):
        return
    authorization = authenticate_request()
    if isinstance(authorization, Response):
        return authorization
    elif not isinstance(authorization, Authorization):
        raise MlflowException(f"Unsupported result type from {auth_config.authorization_function}: '{type(authorization).__name__}'", INTERNAL_ERROR)
    if sender_is_admin():
        return
    if (validator := BEFORE_REQUEST_VALIDATORS.get((request.path, request.method))):
        if not validator():
            return make_forbidden_response()
    elif _is_proxy_artifact_path(request.path):
        if (validator := _get_proxy_artifact_validator(request.method, request.view_args)):
            if not validator():
                return make_forbidden_response()