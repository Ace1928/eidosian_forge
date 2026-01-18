from sqlalchemy import (
from sqlalchemy.orm import backref, relationship
from mlflow.entities.model_registry import (
from mlflow.entities.model_registry.model_version_stages import STAGE_DELETED_INTERNAL, STAGE_NONE
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.store.db.base_sql_model import Base
from mlflow.utils.time import get_current_time_millis
class SqlRegisteredModelTag(Base):
    __tablename__ = 'registered_model_tags'
    name = Column(String(256), ForeignKey('registered_models.name', onupdate='cascade'))
    key = Column(String(250), nullable=False)
    value = Column(String(5000), nullable=True)
    registered_model = relationship('SqlRegisteredModel', backref=backref('registered_model_tags', cascade='all'))
    __table_args__ = (PrimaryKeyConstraint('key', 'name', name='registered_model_tag_pk'),)

    def __repr__(self):
        return f'<SqlRegisteredModelTag ({self.name}, {self.key}, {self.value})>'

    def to_mlflow_entity(self):
        return RegisteredModelTag(self.key, self.value)