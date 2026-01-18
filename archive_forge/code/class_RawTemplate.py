import uuid
from oslo_db.sqlalchemy import models
import sqlalchemy
from sqlalchemy.ext import declarative
from sqlalchemy.orm import backref
from sqlalchemy.orm import relationship
from heat.db import types
class RawTemplate(BASE, HeatBase):
    """Represents an unparsed template which should be in JSON format."""
    __tablename__ = 'raw_template'
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    template = sqlalchemy.Column(types.Json)
    files = sqlalchemy.Column(types.Json)
    files_id = sqlalchemy.Column(sqlalchemy.Integer(), sqlalchemy.ForeignKey('raw_template_files.id'))
    environment = sqlalchemy.Column('environment', types.Json)