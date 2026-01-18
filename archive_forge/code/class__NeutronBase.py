from oslo_db.sqlalchemy import models
from oslo_utils import uuidutils
import sqlalchemy as sa
from sqlalchemy.ext import declarative
from sqlalchemy import orm
from neutron_lib.db import constants as db_const
class _NeutronBase(models.ModelBase):
    """Base class for Neutron Models."""
    __table_args__ = {'mysql_engine': 'InnoDB'}

    def __iter__(self):
        self._i = iter(orm.object_mapper(self).columns)
        return self

    def next(self):
        n = next(self._i).name
        return (n, getattr(self, n))
    __next__ = next

    def __repr__(self):
        """sqlalchemy based automatic __repr__ method."""
        items = ['%s=%r' % (col.name, getattr(self, col.name)) for col in self.__table__.columns]
        return '<%s.%s[object at %x] {%s}>' % (self.__class__.__module__, self.__class__.__name__, id(self), ', '.join(items))