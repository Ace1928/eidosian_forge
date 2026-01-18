from sqlalchemy import (
from sqlalchemy.orm import backref, relationship
from mlflow.entities.model_registry import (
from mlflow.entities.model_registry.model_version_stages import STAGE_DELETED_INTERNAL, STAGE_NONE
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.store.db.base_sql_model import Base
from mlflow.utils.time import get_current_time_millis
class SqlRegisteredModel(Base):
    __tablename__ = 'registered_models'
    name = Column(String(256), unique=True, nullable=False)
    creation_time = Column(BigInteger, default=get_current_time_millis)
    last_updated_time = Column(BigInteger, nullable=True, default=None)
    description = Column(String(5000), nullable=True)
    __table_args__ = (PrimaryKeyConstraint('name', name='registered_model_pk'),)

    def __repr__(self):
        return f'<SqlRegisteredModel ({self.name}, {self.description}, {self.creation_time}, {self.last_updated_time})>'

    def to_mlflow_entity(self):
        latest_versions = {}
        for mv in self.model_versions:
            stage = mv.current_stage
            if stage != STAGE_DELETED_INTERNAL and (stage not in latest_versions or latest_versions[stage].version < mv.version):
                latest_versions[stage] = mv
        return RegisteredModel(self.name, self.creation_time, self.last_updated_time, self.description, [mvd.to_mlflow_entity() for mvd in latest_versions.values()], [tag.to_mlflow_entity() for tag in self.registered_model_tags], [alias.to_mlflow_entity() for alias in self.registered_model_aliases])