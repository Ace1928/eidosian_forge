import uuid
from oslo_db.sqlalchemy import models
from oslo_serialization import jsonutils
from sqlalchemy import BigInteger
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Enum
from sqlalchemy import ForeignKey
from sqlalchemy import Index
from sqlalchemy import Integer
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import backref, relationship
from sqlalchemy import sql
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy.types import TypeDecorator
from sqlalchemy import UniqueConstraint
from glance.common import timeutils
class GlanceBase(models.ModelBase, models.TimestampMixin):
    """Base class for Glance Models."""
    __table_args__ = {'mysql_engine': 'InnoDB', 'mysql_charset': 'utf8'}
    __table_initialized__ = False
    __protected_attributes__ = set(['created_at', 'updated_at', 'deleted_at', 'deleted'])

    def save(self, session=None):
        from glance.db.sqlalchemy import api as db_api
        super(GlanceBase, self).save(session or db_api.get_session())
    created_at = Column(DateTime, default=lambda: timeutils.utcnow(), nullable=False)
    updated_at = Column(DateTime, default=lambda: timeutils.utcnow(), nullable=True, onupdate=lambda: timeutils.utcnow())
    deleted_at = Column(DateTime)
    deleted = Column(Boolean, nullable=False, default=False)

    def delete(self, session=None):
        """Delete this object."""
        self.deleted = True
        self.deleted_at = timeutils.utcnow()
        self.save(session=session)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def to_dict(self):
        d = self.__dict__.copy()
        d.pop('_sa_instance_state')
        return d