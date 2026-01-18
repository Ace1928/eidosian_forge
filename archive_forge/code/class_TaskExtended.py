from datetime import datetime, timezone
import sqlalchemy as sa
from sqlalchemy.types import PickleType
from celery import states
from .session import ResultModelBase
class TaskExtended(Task):
    """For the extend result."""
    __tablename__ = 'celery_taskmeta'
    __table_args__ = {'sqlite_autoincrement': True, 'extend_existing': True}
    name = sa.Column(sa.String(155), nullable=True)
    args = sa.Column(sa.LargeBinary, nullable=True)
    kwargs = sa.Column(sa.LargeBinary, nullable=True)
    worker = sa.Column(sa.String(155), nullable=True)
    retries = sa.Column(sa.Integer, nullable=True)
    queue = sa.Column(sa.String(155), nullable=True)

    def to_dict(self):
        task_dict = super().to_dict()
        task_dict.update({'name': self.name, 'args': self.args, 'kwargs': self.kwargs, 'worker': self.worker, 'retries': self.retries, 'queue': self.queue})
        return task_dict