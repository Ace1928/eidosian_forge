import time
from sqlalchemy import (
from sqlalchemy.orm import backref, declarative_base, relationship
class SqlMetric(Base):
    __tablename__ = 'metrics'
    key = Column(String(250))
    '\n    Metric key: `String` (limit 250 characters). Part of *Primary Key* for ``metrics`` table.\n    '
    value = Column(Float, nullable=False)
    '\n    Metric value: `Float`. Defined as *Non-null* in schema.\n    '
    timestamp = Column(BigInteger, default=lambda: int(time.time()))
    '\n    Timestamp recorded for this metric entry: `BigInteger`. Part of *Primary Key* for\n                                               ``metrics`` table.\n    '
    run_uuid = Column(String(32), ForeignKey('runs.run_uuid'))
    '\n    Run UUID to which this metric belongs to: Part of *Primary Key* for ``metrics`` table.\n                                              *Foreign Key* into ``runs`` table.\n    '
    run = relationship('SqlRun', backref=backref('metrics', cascade='all'))
    '\n    SQLAlchemy relationship (many:one) with :py:class:`mlflow.store.dbmodels.models.SqlRun`.\n    '
    __table_args__ = (PrimaryKeyConstraint('key', 'timestamp', 'run_uuid', name='metric_pk'),)

    def __repr__(self):
        return f'<SqlMetric({self.key}, {self.value}, {self.timestamp})>'