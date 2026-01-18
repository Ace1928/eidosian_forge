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
def _update_latest_metrics_if_necessary(self, logged_metrics, session):

    def _compare_metrics(metric_a, metric_b):
        """
            Returns:
                True if ``metric_a`` is strictly more recent than ``metric_b``, as determined
                by ``step``, ``timestamp``, and ``value``. False otherwise.
            """
        return (metric_a.step, metric_a.timestamp, metric_a.value) > (metric_b.step, metric_b.timestamp, metric_b.value)

    def _overwrite_metric(new_metric, old_metric):
        """
            Writes content of new_metric over old_metric. The content are `value`, `step`,
            `timestamp`, and `is_nan`.

            Returns:
                old_metric with its content updated.
            """
        old_metric.value = new_metric.value
        old_metric.step = new_metric.step
        old_metric.timestamp = new_metric.timestamp
        old_metric.is_nan = new_metric.is_nan
        return old_metric
    if not logged_metrics:
        return
    latest_metrics = {}
    metric_keys = [m.key for m in logged_metrics]
    metric_key_batches = [metric_keys[i:i + 500] for i in range(0, len(metric_keys), 500)]
    for metric_key_batch in metric_key_batches:
        latest_metrics_key_records_from_db = session.query(SqlLatestMetric.key).filter(SqlLatestMetric.run_uuid == logged_metrics[0].run_uuid, SqlLatestMetric.key.in_(metric_key_batch)).all()
        if len(latest_metrics_key_records_from_db) > 0:
            latest_metric_keys_from_db = [record[0] for record in latest_metrics_key_records_from_db]
            latest_metrics_batch = session.query(SqlLatestMetric).filter(SqlLatestMetric.run_uuid == logged_metrics[0].run_uuid, SqlLatestMetric.key.in_(latest_metric_keys_from_db)).order_by(SqlLatestMetric.run_uuid, SqlLatestMetric.key).with_for_update().all()
            latest_metrics.update({m.key: m for m in latest_metrics_batch})
    new_latest_metric_dict = {}
    for logged_metric in logged_metrics:
        latest_metric = latest_metrics.get(logged_metric.key)
        new_latest_metric = new_latest_metric_dict.get(logged_metric.key)
        if not latest_metric and (not new_latest_metric):
            new_latest_metric = SqlLatestMetric(run_uuid=logged_metric.run_uuid, key=logged_metric.key, value=logged_metric.value, timestamp=logged_metric.timestamp, step=logged_metric.step, is_nan=logged_metric.is_nan)
            new_latest_metric_dict[logged_metric.key] = new_latest_metric
        elif not latest_metric and new_latest_metric:
            if _compare_metrics(logged_metric, new_latest_metric):
                new_latest_metric = _overwrite_metric(logged_metric, new_latest_metric)
                new_latest_metric_dict[logged_metric.key] = new_latest_metric
        elif _compare_metrics(logged_metric, latest_metric):
            latest_metric = _overwrite_metric(logged_metric, latest_metric)
    if new_latest_metric_dict:
        session.add_all(new_latest_metric_dict.values())