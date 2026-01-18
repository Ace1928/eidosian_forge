import time
from sqlalchemy import (
from sqlalchemy.orm import backref, declarative_base, relationship
class SqlParam(Base):
    __tablename__ = 'params'
    key = Column(String(250))
    '\n    Param key: `String` (limit 250 characters). Part of *Primary Key* for ``params`` table.\n    '
    value = Column(String(250), nullable=False)
    '\n    Param value: `String` (limit 250 characters). Defined as *Non-null* in schema.\n    '
    run_uuid = Column(String(32), ForeignKey('runs.run_uuid'))
    '\n    Run UUID to which this metric belongs to: Part of *Primary Key* for ``params`` table.\n                                              *Foreign Key* into ``runs`` table.\n    '
    run = relationship('SqlRun', backref=backref('params', cascade='all'))
    '\n    SQLAlchemy relationship (many:one) with :py:class:`mlflow.store.dbmodels.models.SqlRun`.\n    '
    __table_args__ = (PrimaryKeyConstraint('key', 'run_uuid', name='param_pk'),)

    def __repr__(self):
        return f'<SqlParam({self.key}, {self.value})>'