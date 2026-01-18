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
def _get_orderby_clauses(order_by_list, session):
    """Sorts a set of runs based on their natural ordering and an overriding set of order_bys.
    Runs are naturally ordered first by start time descending, then by run id for tie-breaking.
    """
    clauses = []
    ordering_joins = []
    clause_id = 0
    observed_order_by_clauses = set()
    select_clauses = []
    if order_by_list:
        for order_by_clause in order_by_list:
            clause_id += 1
            key_type, key, ascending = SearchUtils.parse_order_by_for_search_runs(order_by_clause)
            key = SearchUtils.translate_key_alias(key)
            if SearchUtils.is_string_attribute(key_type, key, '=') or SearchUtils.is_numeric_attribute(key_type, key, '='):
                order_value = getattr(SqlRun, SqlRun.get_attribute_name(key))
            else:
                if SearchUtils.is_metric(key_type, '='):
                    entity = SqlLatestMetric
                elif SearchUtils.is_tag(key_type, '='):
                    entity = SqlTag
                elif SearchUtils.is_param(key_type, '='):
                    entity = SqlParam
                else:
                    raise MlflowException(f"Invalid identifier type '{key_type}'", error_code=INVALID_PARAMETER_VALUE)
                subquery = session.query(entity).filter(entity.key == key).subquery()
                ordering_joins.append(subquery)
                order_value = subquery.c.value
            if SearchUtils.is_metric(key_type, '='):
                case = sql.case((subquery.c.is_nan == sqlalchemy.true(), 1), (order_value.is_(None), 2), else_=0).label(f'clause_{clause_id}')
            else:
                case = sql.case((order_value.is_(None), 1), else_=0).label(f'clause_{clause_id}')
            clauses.append(case.name)
            select_clauses.append(case)
            select_clauses.append(order_value)
            if (key_type, key) in observed_order_by_clauses:
                raise MlflowException(f'`order_by` contains duplicate fields: {order_by_list}')
            observed_order_by_clauses.add((key_type, key))
            if ascending:
                clauses.append(order_value)
            else:
                clauses.append(order_value.desc())
    if (SearchUtils._ATTRIBUTE_IDENTIFIER, SqlRun.start_time.key) not in observed_order_by_clauses:
        clauses.append(SqlRun.start_time.desc())
    clauses.append(SqlRun.run_uuid)
    return (select_clauses, clauses, ordering_joins)