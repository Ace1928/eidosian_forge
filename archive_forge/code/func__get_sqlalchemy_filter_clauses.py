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
def _get_sqlalchemy_filter_clauses(parsed, session, dialect):
    """
    Creates run attribute filters and subqueries that will be inner-joined to SqlRun to act as
    multi-clause filters and return them as a tuple.
    """
    attribute_filters = []
    non_attribute_filters = []
    dataset_filters = []
    for sql_statement in parsed:
        key_type = sql_statement.get('type')
        key_name = sql_statement.get('key')
        value = sql_statement.get('value')
        comparator = sql_statement.get('comparator').upper()
        key_name = SearchUtils.translate_key_alias(key_name)
        if SearchUtils.is_string_attribute(key_type, key_name, comparator) or SearchUtils.is_numeric_attribute(key_type, key_name, comparator):
            if key_name == 'run_name':
                key_filter = SearchUtils.get_sql_comparison_func('=', dialect)(SqlTag.key, MLFLOW_RUN_NAME)
                val_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(SqlTag.value, value)
                non_attribute_filters.append(session.query(SqlTag).filter(key_filter, val_filter).subquery())
            else:
                attribute = getattr(SqlRun, SqlRun.get_attribute_name(key_name))
                attr_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(attribute, value)
                attribute_filters.append(attr_filter)
        else:
            if SearchUtils.is_metric(key_type, comparator):
                entity = SqlLatestMetric
                value = float(value)
            elif SearchUtils.is_param(key_type, comparator):
                entity = SqlParam
            elif SearchUtils.is_tag(key_type, comparator):
                entity = SqlTag
            elif SearchUtils.is_dataset(key_type, comparator):
                entity = SqlDataset
            else:
                raise MlflowException(f"Invalid search expression type '{key_type}'", error_code=INVALID_PARAMETER_VALUE)
            if entity == SqlDataset:
                if key_name == 'context':
                    dataset_filters.append(session.query(entity, SqlInput, SqlInputTag).join(SqlInput, SqlInput.source_id == SqlDataset.dataset_uuid).join(SqlInputTag, and_(SqlInputTag.input_uuid == SqlInput.input_uuid, SqlInputTag.name == MLFLOW_DATASET_CONTEXT, SearchUtils.get_sql_comparison_func(comparator, dialect)(getattr(SqlInputTag, 'value'), value))).subquery())
                else:
                    dataset_attr_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(getattr(SqlDataset, key_name), value)
                    dataset_filters.append(session.query(entity, SqlInput).join(SqlInput, SqlInput.source_id == SqlDataset.dataset_uuid).filter(dataset_attr_filter).subquery())
            else:
                key_filter = SearchUtils.get_sql_comparison_func('=', dialect)(entity.key, key_name)
                val_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(entity.value, value)
                non_attribute_filters.append(session.query(entity).filter(key_filter, val_filter).subquery())
    return (attribute_filters, non_attribute_filters, dataset_filters)