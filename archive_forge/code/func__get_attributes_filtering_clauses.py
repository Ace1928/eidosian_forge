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
def _get_attributes_filtering_clauses(parsed, dialect):
    clauses = []
    for sql_statement in parsed:
        key_type = sql_statement.get('type')
        key_name = sql_statement.get('key')
        value = sql_statement.get('value')
        comparator = sql_statement.get('comparator').upper()
        if SearchUtils.is_string_attribute(key_type, key_name, comparator) or SearchUtils.is_numeric_attribute(key_type, key_name, comparator):
            attribute = getattr(SqlRun, SqlRun.get_attribute_name(key_name))
            clauses.append(SearchUtils.get_sql_comparison_func(comparator, dialect)(attribute, value))
    return clauses