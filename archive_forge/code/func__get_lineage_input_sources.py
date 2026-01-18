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
def _get_lineage_input_sources(self, run):
    from mlflow.data.delta_dataset_source import DeltaDatasetSource
    if run is None:
        return None
    securable_list = []
    if run.inputs is not None:
        for dataset in run.inputs.dataset_inputs:
            dataset_source = mlflow.data.get_source(dataset)
            if isinstance(dataset_source, DeltaDatasetSource) and dataset_source._get_source_type() == _DELTA_TABLE:
                if dataset_source.delta_table_name and dataset_source.delta_table_id:
                    table_entity = Table(name=dataset_source.delta_table_name, table_id=dataset_source.delta_table_id)
                    securable_list.append(Securable(table=table_entity))
        if len(securable_list) > _MAX_LINEAGE_DATA_SOURCES:
            _logger.warning(f'Model version has {len(securable_list)!s} upstream datasets, which exceeds the max of 10 upstream datasets for lineage tracking. Only the first 10 datasets will be propagated to Unity Catalog lineage')
        return securable_list[0:_MAX_LINEAGE_DATA_SOURCES]
    else:
        return None