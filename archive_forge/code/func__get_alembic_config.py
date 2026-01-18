import logging
import os
import time
from contextlib import contextmanager
import sqlalchemy
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import sql
from sqlalchemy.pool import (
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INTERNAL_ERROR, TEMPORARILY_UNAVAILABLE
from mlflow.store.db.db_types import SQLITE
from mlflow.store.model_registry.dbmodels.models import (
from mlflow.store.tracking.dbmodels.initial_models import Base as InitialBase
from mlflow.store.tracking.dbmodels.models import (
def _get_alembic_config(db_url, alembic_dir=None):
    """
    Constructs an alembic Config object referencing the specified database and migration script
    directory.

    Args:
        db_url: Database URL, like sqlite:///<absolute-path-to-local-db-file>. See
            https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls for a full list of
            valid database URLs.
        alembic_dir: Path to migration script directory. Uses canonical migration script
            directory under mlflow/alembic if unspecified. TODO: remove this argument in MLflow 1.1,
            as it's only used to run special migrations for pre-1.0 users to remove duplicate
            constraint names.
    """
    from alembic.config import Config
    final_alembic_dir = os.path.join(_get_package_dir(), 'store', 'db_migrations') if alembic_dir is None else alembic_dir
    db_url = db_url.replace('%', '%%')
    config = Config(os.path.join(final_alembic_dir, 'alembic.ini'))
    config.set_main_option('script_location', final_alembic_dir)
    config.set_main_option('sqlalchemy.url', db_url)
    return config