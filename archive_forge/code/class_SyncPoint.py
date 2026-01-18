import uuid
from oslo_db.sqlalchemy import models
import sqlalchemy
from sqlalchemy.ext import declarative
from sqlalchemy.orm import backref
from sqlalchemy.orm import relationship
from heat.db import types
class SyncPoint(BASE, HeatBase):
    """Represents a syncpoint for a stack that is being worked on."""
    __tablename__ = 'sync_point'
    __table_args__ = (sqlalchemy.PrimaryKeyConstraint('entity_id', 'traversal_id', 'is_update'), sqlalchemy.ForeignKeyConstraint(['stack_id'], ['stack.id']))
    entity_id = sqlalchemy.Column(sqlalchemy.String(36))
    traversal_id = sqlalchemy.Column(sqlalchemy.String(36))
    is_update = sqlalchemy.Column(sqlalchemy.Boolean)
    atomic_key = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    stack_id = sqlalchemy.Column(sqlalchemy.String(36), nullable=False)
    input_data = sqlalchemy.Column(types.Json)