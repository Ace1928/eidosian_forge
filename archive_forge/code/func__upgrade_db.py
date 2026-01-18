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
def _upgrade_db(engine):
    """
    Upgrade the schema of an MLflow tracking database to the latest supported version.
    Note that schema migrations can be slow and are not guaranteed to be transactional -
    we recommend taking a backup of your database before running migrations.

    Args:
        url: Database URL, like sqlite:///<absolute-path-to-local-db-file>. See
            https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls for a full list of
            valid database URLs.
    """
    from alembic import command
    db_url = str(engine.url)
    _logger.info('Updating database tables')
    config = _get_alembic_config(db_url)
    with engine.begin() as connection:
        config.attributes['connection'] = connection
        command.upgrade(config, 'heads')