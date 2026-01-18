import sqlalchemy as sa
from sqlalchemy import (
from sqlalchemy.orm import backref, relationship
from mlflow.entities import (
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.store.db.base_sql_model import Base
from mlflow.utils.mlflow_tags import _get_run_name_from_tags
from mlflow.utils.time import get_current_time_millis
class SqlInputTag(Base):
    __tablename__ = 'input_tags'
    __table_args__ = (PrimaryKeyConstraint('input_uuid', 'name', name='input_tags_pk'),)
    input_uuid = Column(String(36), ForeignKey('inputs.input_uuid'), nullable=False)
    '\n    Input UUID: `String` (limit 36 characters). Defined as *Non-null* in schema.\n    *Foreign Key* into ``inputs`` table. Part of *Primary Key* for ``input_tags`` table.\n    '
    name = Column(String(255), nullable=False)
    '\n    Param name: `String` (limit 255 characters). Defined as *Non-null* in schema.\n    Part of *Primary Key* for ``input_tags`` table.\n    '
    value = Column(String(500), nullable=False)
    '\n    Param value: `String` (limit 500 characters). Defined as *Non-null* in schema.\n    Part of *Primary Key* for ``input_tags`` table.\n    '

    def __repr__(self):
        return f'<SqlInputTag ({self.input_uuid}, {self.name}, {self.value})>'

    def to_mlflow_entity(self):
        """
        Convert DB model to corresponding MLflow entity.

        Returns:
            mlflow.entities.InputTag: Description of the return value.
        """
        return InputTag(key=self.name, value=self.value)