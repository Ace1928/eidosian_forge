from sqlalchemy import (
from sqlalchemy.orm import declarative_base, relationship
from mlflow.server.auth.entities import ExperimentPermission, RegisteredModelPermission, User
class SqlRegisteredModelPermission(Base):
    __tablename__ = 'registered_model_permissions'
    id = Column(Integer(), primary_key=True)
    name = Column(String(255), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    permission = Column(String(255))
    __table_args__ = (UniqueConstraint('name', 'user_id', name='unique_name_user'),)

    def to_mlflow_entity(self):
        return RegisteredModelPermission(name=self.name, user_id=self.user_id, permission=self.permission)