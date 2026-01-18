import sqlalchemy as sa
from sqlalchemy import (
from sqlalchemy.orm import backref, relationship
from mlflow.entities import (
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.store.db.base_sql_model import Base
from mlflow.utils.mlflow_tags import _get_run_name_from_tags
from mlflow.utils.time import get_current_time_millis
class SqlExperimentTag(Base):
    """
    DB model for :py:class:`mlflow.entities.RunTag`.
    These are recorded in ``experiment_tags`` table.
    """
    __tablename__ = 'experiment_tags'
    key = Column(String(250))
    '\n    Tag key: `String` (limit 250 characters). *Primary Key* for ``tags`` table.\n    '
    value = Column(String(5000), nullable=True)
    '\n    Value associated with tag: `String` (limit 5000 characters). Could be *null*.\n    '
    experiment_id = Column(Integer, ForeignKey('experiments.experiment_id'))
    '\n    Experiment ID to which this tag belongs: *Foreign Key* into ``experiments`` table.\n    '
    experiment = relationship('SqlExperiment', backref=backref('tags', cascade='all'))
    '\n    SQLAlchemy relationship (many:one) with :py:class:`mlflow.store.dbmodels.models.SqlExperiment`.\n    '
    __table_args__ = (PrimaryKeyConstraint('key', 'experiment_id', name='experiment_tag_pk'),)

    def __repr__(self):
        return f'<SqlExperimentTag({self.key}, {self.value})>'

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        Returns:
            mlflow.entities.RunTag: Description of the return value.
        """
        return ExperimentTag(key=self.key, value=self.value)