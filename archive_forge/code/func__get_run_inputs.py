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
def _get_run_inputs(self, session, run_uuids):
    datasets = session.query(SqlInput.input_uuid, SqlInput.destination_id.label('run_uuid'), SqlDataset).select_from(SqlDataset).join(SqlInput, SqlInput.source_id == SqlDataset.dataset_uuid).filter(SqlInput.destination_type == 'RUN', SqlInput.destination_id.in_(run_uuids)).order_by('run_uuid').all()
    input_uuids = [dataset.input_uuid for dataset in datasets]
    input_tags = session.query(SqlInput.input_uuid, SqlInput.destination_id.label('run_uuid'), SqlInputTag).join(SqlInput, SqlInput.input_uuid == SqlInputTag.input_uuid).filter(SqlInput.input_uuid.in_(input_uuids)).order_by('run_uuid').all()
    all_dataset_inputs = []
    for run_uuid in run_uuids:
        dataset_inputs = []
        for input_uuid, dataset_run_uuid, dataset_sql in datasets:
            if run_uuid == dataset_run_uuid:
                dataset_entity = dataset_sql.to_mlflow_entity()
                tags = []
                for tag_input_uuid, tag_run_uuid, tag_sql in input_tags:
                    if input_uuid == tag_input_uuid and run_uuid == tag_run_uuid:
                        tags.append(tag_sql.to_mlflow_entity())
                dataset_input_entity = DatasetInput(dataset=dataset_entity, tags=tags)
                dataset_inputs.append(dataset_input_entity)
        all_dataset_inputs.append(dataset_inputs)
    return all_dataset_inputs