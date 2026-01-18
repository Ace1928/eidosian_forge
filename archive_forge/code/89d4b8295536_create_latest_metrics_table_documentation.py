import time
import logging
from alembic import op
from sqlalchemy import orm, func, distinct, and_
from sqlalchemy import Column, String, ForeignKey, Float, BigInteger, PrimaryKeyConstraint, Boolean
from mlflow.store.tracking.dbmodels.models import SqlMetric, SqlLatestMetric

    If the targeted database contains any metric entries, this function emits important,
    database-specific information about the ``create_latest_metrics_table`` migration.
    If the targeted database does *not* contain any metric entries, this output is omitted
    in order to avoid superfluous log output when initializing a new Tracking database.
    