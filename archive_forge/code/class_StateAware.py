import uuid
from oslo_db.sqlalchemy import models
import sqlalchemy
from sqlalchemy.ext import declarative
from sqlalchemy.orm import backref
from sqlalchemy.orm import relationship
from heat.db import types
class StateAware(object):
    action = sqlalchemy.Column('action', sqlalchemy.String(255))
    status = sqlalchemy.Column('status', sqlalchemy.String(255))
    status_reason = sqlalchemy.Column('status_reason', sqlalchemy.Text)