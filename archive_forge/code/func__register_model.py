from typing import Any, Dict, List, Optional
from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ALREADY_EXISTS, RESOURCE_ALREADY_EXISTS, ErrorCode
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.store.model_registry import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.client import MlflowClient
from mlflow.utils import get_results_from_paginated_fn
from mlflow.utils.logging_utils import eprint
def _register_model(model_uri, name, await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS, *, tags: Optional[Dict[str, Any]]=None, local_model_path=None) -> ModelVersion:
    client = MlflowClient()
    try:
        create_model_response = client.create_registered_model(name)
        eprint(f"Successfully registered model '{create_model_response.name}'.")
    except MlflowException as e:
        if e.error_code in (ErrorCode.Name(RESOURCE_ALREADY_EXISTS), ErrorCode.Name(ALREADY_EXISTS)):
            eprint("Registered model '%s' already exists. Creating a new version of this model..." % name)
        else:
            raise e
    run_id = None
    source = model_uri
    if RunsArtifactRepository.is_runs_uri(model_uri):
        source = RunsArtifactRepository.get_underlying_uri(model_uri)
        run_id, _ = RunsArtifactRepository.parse_runs_uri(model_uri)
    create_version_response = client._create_model_version(name=name, source=source, run_id=run_id, tags=tags, await_creation_for=await_registration_for, local_model_path=local_model_path)
    eprint(f"Created version '{create_version_response.version}' of model '{create_version_response.name}'.")
    return create_version_response