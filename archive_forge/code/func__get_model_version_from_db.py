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
def _get_model_version_from_db(cls, session, name, version, conditions, query_options=None):
    if query_options is None:
        query_options = []
    versions = session.query(SqlModelVersion).options(*query_options).filter(*conditions).all()
    if len(versions) == 0:
        raise MlflowException(f'Model Version (name={name}, version={version}) not found', RESOURCE_DOES_NOT_EXIST)
    if len(versions) > 1:
        raise MlflowException(f'Expected only 1 model version with (name={name}, version={version}). Found {len(versions)}.', INVALID_STATE)
    return versions[0]