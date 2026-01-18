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
class CachedImages(BASE, models.ModelBase):
    """Represents an image tag in the datastore."""
    __tablename__ = 'cached_images'
    __table_args__ = (UniqueConstraint('image_id', 'node_reference_id', name='ix_cached_images_image_id_node_reference_id'),)
    id = Column(BigInteger().with_variant(Integer, 'sqlite'), primary_key=True, autoincrement=True, nullable=False)
    image_id = Column(String(36), nullable=False)
    last_accessed = Column(DateTime, nullable=False)
    last_modified = Column(DateTime, nullable=False)
    size = Column(BigInteger(), nullable=False)
    hits = Column(Integer, nullable=False)
    checksum = Column(String(32), nullable=True)
    node_reference_id = Column(BigInteger().with_variant(Integer, 'sqlite'), ForeignKey('node_reference.node_reference_id'), nullable=False)