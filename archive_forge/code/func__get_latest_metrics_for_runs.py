import time
import logging
from alembic import op
from sqlalchemy import orm, func, distinct, and_
from sqlalchemy import Column, String, ForeignKey, Float, BigInteger, PrimaryKeyConstraint, Boolean
from mlflow.store.tracking.dbmodels.models import SqlMetric, SqlLatestMetric
def _get_latest_metrics_for_runs(session):
    metrics_with_max_step = session.query(SqlMetric.run_uuid, SqlMetric.key, func.max(SqlMetric.step).label('step')).group_by(SqlMetric.key, SqlMetric.run_uuid).subquery('metrics_with_max_step')
    metrics_with_max_timestamp = session.query(SqlMetric.run_uuid, SqlMetric.key, SqlMetric.step, func.max(SqlMetric.timestamp).label('timestamp')).join(metrics_with_max_step, and_(SqlMetric.step == metrics_with_max_step.c.step, SqlMetric.run_uuid == metrics_with_max_step.c.run_uuid, SqlMetric.key == metrics_with_max_step.c.key)).group_by(SqlMetric.key, SqlMetric.run_uuid, SqlMetric.step).subquery('metrics_with_max_timestamp')
    metrics_with_max_value = session.query(SqlMetric.run_uuid, SqlMetric.key, SqlMetric.step, SqlMetric.timestamp, func.max(SqlMetric.value).label('value'), SqlMetric.is_nan).join(metrics_with_max_timestamp, and_(SqlMetric.timestamp == metrics_with_max_timestamp.c.timestamp, SqlMetric.run_uuid == metrics_with_max_timestamp.c.run_uuid, SqlMetric.key == metrics_with_max_timestamp.c.key, SqlMetric.step == metrics_with_max_timestamp.c.step)).group_by(SqlMetric.run_uuid, SqlMetric.key, SqlMetric.step, SqlMetric.timestamp, SqlMetric.is_nan).all()
    return metrics_with_max_value