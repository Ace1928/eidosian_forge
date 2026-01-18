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
class NodeReference(BASE, models.ModelBase):
    """Represents node info in the datastore"""
    __tablename__ = 'node_reference'
    __table_args__ = (UniqueConstraint('node_reference_url', name='uq_node_reference_node_reference_url'),)
    node_reference_id = Column(BigInteger().with_variant(Integer, 'sqlite'), primary_key=True, nullable=False, autoincrement=True)
    node_reference_url = Column(String(length=255), nullable=False)