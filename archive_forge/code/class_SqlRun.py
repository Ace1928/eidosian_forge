import time
from sqlalchemy import (
from sqlalchemy.orm import backref, declarative_base, relationship
class SqlRun(Base):
    """
    DB model for :py:class:`mlflow.entities.Run`. These are recorded in ``runs`` table.
    """
    __tablename__ = 'runs'
    run_uuid = Column(String(32), nullable=False)
    '\n    Run UUID: `String` (limit 32 characters). *Primary Key* for ``runs`` table.\n    '
    name = Column(String(250))
    '\n    Run name: `String` (limit 250 characters).\n    '
    source_type = Column(String(20), default='LOCAL')
    '\n    Source Type: `String` (limit 20 characters). Can be one of ``NOTEBOOK``, ``JOB``, ``PROJECT``,\n                 ``LOCAL`` (default), or ``UNKNOWN``.\n    '
    source_name = Column(String(500))
    '\n    Name of source recording the run: `String` (limit 500 characters).\n    '
    entry_point_name = Column(String(50))
    '\n    Entry-point name that launched the run run: `String` (limit 50 characters).\n    '
    user_id = Column(String(256), nullable=True, default=None)
    '\n    User ID: `String` (limit 256 characters). Defaults to ``null``.\n    '
    status = Column(String(20), default='SCHEDULED')
    '\n    Run Status: `String` (limit 20 characters). Can be one of ``RUNNING``, ``SCHEDULED`` (default),\n                ``FINISHED``, ``FAILED``.\n    '
    start_time = Column(BigInteger, default=int(time.time()))
    '\n    Run start time: `BigInteger`. Defaults to current system time.\n    '
    end_time = Column(BigInteger, nullable=True, default=None)
    '\n    Run end time: `BigInteger`.\n    '
    source_version = Column(String(50))
    '\n    Source version: `String` (limit 50 characters).\n    '
    lifecycle_stage = Column(String(20), default='active')
    '\n    Lifecycle Stage of run: `String` (limit 32 characters).\n                            Can be either ``active`` (default) or ``deleted``.\n    '
    artifact_uri = Column(String(200), default=None)
    '\n    Default artifact location for this run: `String` (limit 200 characters).\n    '
    experiment_id = Column(Integer, ForeignKey('experiments.experiment_id'))
    '\n    Experiment ID to which this run belongs to: *Foreign Key* into ``experiment`` table.\n    '
    experiment = relationship('SqlExperiment', backref=backref('runs', cascade='all'))
    '\n    SQLAlchemy relationship (many:one) with :py:class:`mlflow.store.dbmodels.models.SqlExperiment`.\n    '
    __table_args__ = (CheckConstraint(source_type.in_(SourceTypes), name='source_type'), CheckConstraint(status.in_(RunStatusTypes), name='status'), CheckConstraint(lifecycle_stage.in_(['active', 'deleted']), name='runs_lifecycle_stage'), PrimaryKeyConstraint('run_uuid', name='run_pk'))