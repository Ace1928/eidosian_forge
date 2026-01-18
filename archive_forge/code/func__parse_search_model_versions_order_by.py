import logging
import urllib
import sqlalchemy
from sqlalchemy.future import select
import mlflow.store.db.utils
from mlflow.entities.model_registry.model_version_stages import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
from mlflow.store.artifact.utils.models import _parse_model_uri
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry import (
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.store.model_registry.dbmodels.models import (
from mlflow.utils.search_utils import SearchModelUtils, SearchModelVersionUtils, SearchUtils
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import extract_db_type_from_uri
from mlflow.utils.validation import (
@classmethod
def _parse_search_model_versions_order_by(cls, order_by_list):
    """Sorts a set of model versions based on their natural ordering and an overriding set
        of order_bys. Model versions are naturally ordered first by name ascending, then by
        version ascending.
        """
    clauses = []
    observed_order_by_clauses = set()
    if order_by_list:
        for order_by_clause in order_by_list:
            _, key, ascending = SearchModelVersionUtils.parse_order_by_for_search_model_versions(order_by_clause)
            if key not in SearchModelVersionUtils.VALID_ORDER_BY_ATTRIBUTE_KEYS:
                raise MlflowException(f"Invalid order by key '{key}' specified. Valid keys are {SearchModelVersionUtils.VALID_ORDER_BY_ATTRIBUTE_KEYS}", error_code=INVALID_PARAMETER_VALUE)
            elif key == 'version_number':
                field = SqlModelVersion.version
            elif key == 'creation_timestamp':
                field = SqlModelVersion.creation_time
            elif key == 'last_updated_timestamp':
                field = SqlModelVersion.last_updated_time
            else:
                field = getattr(SqlModelVersion, key)
            if field.key in observed_order_by_clauses:
                raise MlflowException(f'`order_by` contains duplicate fields: {order_by_list}')
            observed_order_by_clauses.add(field.key)
            if ascending:
                clauses.append(field.asc())
            else:
                clauses.append(field.desc())
    if SqlModelVersion.name.key not in observed_order_by_clauses:
        clauses.append(SqlModelVersion.name.asc())
    if SqlModelVersion.version.key not in observed_order_by_clauses:
        clauses.append(SqlModelVersion.version.desc())
    return clauses