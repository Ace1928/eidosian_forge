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
def _get_registered_model(cls, session, name, eager=False):
    """
        Args:
            eager: If ``True``, eagerly loads the registered model's tags. If ``False``, these
                attributes are not eagerly loaded and will be loaded when their corresponding object
                properties are accessed from the resulting ``SqlRegisteredModel`` object.
        """
    _validate_model_name(name)
    query_options = cls._get_eager_registered_model_query_options() if eager else []
    rms = session.query(SqlRegisteredModel).options(*query_options).filter(SqlRegisteredModel.name == name).all()
    if len(rms) == 0:
        raise MlflowException(f'Registered Model with name={name} not found', RESOURCE_DOES_NOT_EXIST)
    if len(rms) > 1:
        raise MlflowException(f'Expected only 1 registered model with name={name}. Found {len(rms)}.', INVALID_STATE)
    return rms[0]