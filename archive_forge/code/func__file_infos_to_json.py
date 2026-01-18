import logging
import click
from mlflow.artifacts import download_artifacts as _download_artifacts
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.tracking import _get_store
from mlflow.utils.proto_json_utils import message_to_json
def _file_infos_to_json(file_infos):
    json_list = [message_to_json(file_info.to_proto()) for file_info in file_infos]
    return '[' + ', '.join(json_list) + ']'