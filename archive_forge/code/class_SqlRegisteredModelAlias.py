from sqlalchemy import (
from sqlalchemy.orm import backref, relationship
from mlflow.entities.model_registry import (
from mlflow.entities.model_registry.model_version_stages import STAGE_DELETED_INTERNAL, STAGE_NONE
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.store.db.base_sql_model import Base
from mlflow.utils.time import get_current_time_millis
class SqlRegisteredModelAlias(Base):
    __tablename__ = 'registered_model_aliases'
    name = Column(String(256), ForeignKey('registered_models.name', onupdate='cascade', ondelete='cascade', name='registered_model_alias_name_fkey'))
    alias = Column(String(256), nullable=False)
    version = Column(Integer, nullable=False)
    registered_model = relationship('SqlRegisteredModel', backref=backref('registered_model_aliases', cascade='all'))
    __table_args__ = (PrimaryKeyConstraint('name', 'alias', name='registered_model_alias_pk'),)

    def __repr__(self):
        return f'<SqlRegisteredModelAlias ({self.name}, {self.alias}, {self.version})>'

    def to_mlflow_entity(self):
        return RegisteredModelAlias(self.alias, self.version)