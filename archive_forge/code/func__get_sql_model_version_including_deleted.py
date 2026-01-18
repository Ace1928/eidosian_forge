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
def _get_sql_model_version_including_deleted(self, name, version):
    """
        Private method to retrieve model versions including those that are internally deleted.
        Used in tests to verify redaction behavior on deletion.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
    with self.ManagedSessionMaker() as session:
        conditions = [SqlModelVersion.name == name, SqlModelVersion.version == version]
        sql_model_version = self._get_model_version_from_db(session, name, version, conditions)
        return self._populate_model_version_aliases(session, name, sql_model_version.to_mlflow_entity())