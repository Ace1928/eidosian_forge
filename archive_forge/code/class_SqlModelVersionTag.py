from sqlalchemy import (
from sqlalchemy.orm import backref, relationship
from mlflow.entities.model_registry import (
from mlflow.entities.model_registry.model_version_stages import STAGE_DELETED_INTERNAL, STAGE_NONE
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.store.db.base_sql_model import Base
from mlflow.utils.time import get_current_time_millis
class SqlModelVersionTag(Base):
    __tablename__ = 'model_version_tags'
    name = Column(String(256))
    version = Column(Integer)
    key = Column(String(250), nullable=False)
    value = Column(String(5000), nullable=True)
    model_version = relationship('SqlModelVersion', foreign_keys=[name, version], backref=backref('model_version_tags', cascade='all'))
    __table_args__ = (PrimaryKeyConstraint('key', 'name', 'version', name='model_version_tag_pk'), ForeignKeyConstraint(('name', 'version'), ('model_versions.name', 'model_versions.version'), onupdate='cascade'))

    def __repr__(self):
        return f'<SqlModelVersionTag ({self.name}, {self.version}, {self.key}, {self.value})>'

    def to_mlflow_entity(self):
        return ModelVersionTag(self.key, self.value)