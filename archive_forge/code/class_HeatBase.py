import uuid
from oslo_db.sqlalchemy import models
import sqlalchemy
from sqlalchemy.ext import declarative
from sqlalchemy.orm import backref
from sqlalchemy.orm import relationship
from heat.db import types
class HeatBase(models.ModelBase, models.TimestampMixin):
    """Base class for Heat Models."""
    __table_args__ = {'mysql_engine': 'InnoDB'}