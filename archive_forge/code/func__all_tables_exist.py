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
def _all_tables_exist(engine):
    return {t for t in sqlalchemy.inspect(engine).get_table_names() if not t.startswith('alembic_')} == {SqlExperiment.__tablename__, SqlRun.__tablename__, SqlMetric.__tablename__, SqlParam.__tablename__, SqlTag.__tablename__, SqlExperimentTag.__tablename__, SqlLatestMetric.__tablename__, SqlRegisteredModel.__tablename__, SqlModelVersion.__tablename__, SqlRegisteredModelTag.__tablename__, SqlModelVersionTag.__tablename__, SqlRegisteredModelAlias.__tablename__, SqlDataset.__tablename__, SqlInput.__tablename__, SqlInputTag.__tablename__}