import json
import logging
import math
import random
import threading
import time
import uuid
from functools import reduce
from typing import List, Optional
import sqlalchemy
import sqlalchemy.sql.expression as sql
from sqlalchemy import and_, func, sql, text
from sqlalchemy.future import select
import mlflow.store.db.utils
from mlflow.entities import (
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities.metric import MetricWithRunId
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
from mlflow.store.db.db_types import MSSQL, MYSQL
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT, SEARCH_MAX_RESULTS_THRESHOLD
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking.dbmodels.models import (
from mlflow.utils.file_utils import local_file_uri_to_path, mkdir
from mlflow.utils.mlflow_tags import (
from mlflow.utils.name_utils import _generate_random_name
from mlflow.utils.search_utils import SearchExperimentsUtils, SearchUtils
from mlflow.utils.string_utils import is_string_type
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import (
from mlflow.utils.validation import (
def _log_inputs_impl(self, experiment_id, run_id, dataset_inputs: Optional[List[DatasetInput]]=None):
    if dataset_inputs is None or len(dataset_inputs) == 0:
        return
    for dataset_input in dataset_inputs:
        if dataset_input.dataset is None:
            raise MlflowException('Dataset input must have a dataset associated with it.', INTERNAL_ERROR)
    name_digest_keys = {}
    for dataset_input in dataset_inputs:
        key = (dataset_input.dataset.name, dataset_input.dataset.digest)
        if key not in name_digest_keys:
            name_digest_keys[key] = dataset_input
    dataset_inputs = list(name_digest_keys.values())
    with self.ManagedSessionMaker() as session:
        dataset_names_to_check = [dataset_input.dataset.name for dataset_input in dataset_inputs]
        dataset_digests_to_check = [dataset_input.dataset.digest for dataset_input in dataset_inputs]
        existing_datasets = session.query(SqlDataset).filter(SqlDataset.name.in_(dataset_names_to_check)).filter(SqlDataset.digest.in_(dataset_digests_to_check)).all()
        dataset_uuids = {}
        for existing_dataset in existing_datasets:
            dataset_uuids[existing_dataset.name, existing_dataset.digest] = existing_dataset.dataset_uuid
        objs_to_write = []
        for dataset_input in dataset_inputs:
            if (dataset_input.dataset.name, dataset_input.dataset.digest) not in dataset_uuids:
                new_dataset_uuid = uuid.uuid4().hex
                dataset_uuids[dataset_input.dataset.name, dataset_input.dataset.digest] = new_dataset_uuid
                objs_to_write.append(SqlDataset(dataset_uuid=new_dataset_uuid, experiment_id=experiment_id, name=dataset_input.dataset.name, digest=dataset_input.dataset.digest, dataset_source_type=dataset_input.dataset.source_type, dataset_source=dataset_input.dataset.source, dataset_schema=dataset_input.dataset.schema, dataset_profile=dataset_input.dataset.profile))
        existing_inputs = session.query(SqlInput).filter(SqlInput.source_type == 'DATASET').filter(SqlInput.source_id.in_(dataset_uuids.values())).filter(SqlInput.destination_type == 'RUN').filter(SqlInput.destination_id == run_id).all()
        input_uuids = {}
        for existing_input in existing_inputs:
            input_uuids[existing_input.source_id, existing_input.destination_id] = existing_input.input_uuid
        for dataset_input in dataset_inputs:
            dataset_uuid = dataset_uuids[dataset_input.dataset.name, dataset_input.dataset.digest]
            if (dataset_uuid, run_id) not in input_uuids:
                new_input_uuid = uuid.uuid4().hex
                input_uuids[dataset_input.dataset.name, dataset_input.dataset.digest] = new_input_uuid
                objs_to_write.append(SqlInput(input_uuid=new_input_uuid, source_type='DATASET', source_id=dataset_uuid, destination_type='RUN', destination_id=run_id))
                for input_tag in dataset_input.tags:
                    objs_to_write.append(SqlInputTag(input_uuid=new_input_uuid, name=input_tag.key, value=input_tag.value))
        session.add_all(objs_to_write)