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
def _get_latest_schema_revision():
    """Get latest schema revision as a string."""
    config = _get_alembic_config(db_url='')
    script = ScriptDirectory.from_config(config)
    heads = script.get_heads()
    if len(heads) != 1:
        raise MlflowException(f'Migration script directory was in unexpected state. Got {len(heads)} head database versions but expected only 1. Found versions: {heads}')
    return heads[0]