from oslo_utils import timeutils
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy.orm import object_mapper
from oslo_db.sqlalchemy import types
class TimestampMixin(object):
    created_at = Column(DateTime, default=lambda: timeutils.utcnow())
    updated_at = Column(DateTime, onupdate=lambda: timeutils.utcnow())