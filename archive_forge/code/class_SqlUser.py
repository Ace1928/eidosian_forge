from sqlalchemy import (
from sqlalchemy.orm import declarative_base, relationship
from mlflow.server.auth.entities import ExperimentPermission, RegisteredModelPermission, User
class SqlUser(Base):
    __tablename__ = 'users'
    id = Column(Integer(), primary_key=True)
    username = Column(String(255), unique=True)
    password_hash = Column(String(255))
    is_admin = Column(Boolean, default=False)
    experiment_permissions = relationship('SqlExperimentPermission', backref='users')
    registered_model_permissions = relationship('SqlRegisteredModelPermission', backref='users')

    def to_mlflow_entity(self):
        return User(id_=self.id, username=self.username, password_hash=self.password_hash, is_admin=self.is_admin, experiment_permissions=[p.to_mlflow_entity() for p in self.experiment_permissions], registered_model_permissions=[p.to_mlflow_entity() for p in self.registered_model_permissions])