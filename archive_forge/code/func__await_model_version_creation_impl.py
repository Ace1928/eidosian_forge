import logging
from abc import ABCMeta, abstractmethod
from time import sleep, time
from mlflow.entities.model_registry import ModelVersionTag
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS, ErrorCode
from mlflow.utils.annotations import developer_stable
def _await_model_version_creation_impl(self, mv, await_creation_for, hint=''):
    _logger.info(f'Waiting up to {await_creation_for} seconds for model version to finish creation. Model name: {mv.name}, version {mv.version}')
    max_time = time() + await_creation_for
    pending_status = ModelVersionStatus.to_string(ModelVersionStatus.PENDING_REGISTRATION)
    while mv.status == pending_status:
        if time() > max_time:
            raise MlflowException(f'Exceeded max wait time for model name: {mv.name} version: {mv.version} to become READY. Status: {mv.status} Wait Time: {await_creation_for}.{hint}')
        mv = self.get_model_version(mv.name, mv.version)
        sleep(AWAIT_MODEL_VERSION_CREATE_SLEEP_INTERVAL_SECONDS)
    if mv.status != ModelVersionStatus.to_string(ModelVersionStatus.READY):
        raise MlflowException(f'Model version creation failed for model name: {mv.name} version: {mv.version} with status: {mv.status} and message: {mv.status_message}')