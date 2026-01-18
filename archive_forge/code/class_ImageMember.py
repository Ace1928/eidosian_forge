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
class ImageMember(BASE, GlanceBase):
    """Represents an image members in the datastore."""
    __tablename__ = 'image_members'
    unique_constraint_key_name = 'image_members_image_id_member_deleted_at_key'
    __table_args__ = (Index('ix_image_members_deleted', 'deleted'), Index('ix_image_members_image_id', 'image_id'), Index('ix_image_members_image_id_member', 'image_id', 'member'), UniqueConstraint('image_id', 'member', 'deleted_at', name=unique_constraint_key_name))
    id = Column(Integer, primary_key=True)
    image_id = Column(String(36), ForeignKey('images.id'), nullable=False)
    image = relationship(Image, backref=backref('members'))
    member = Column(String(255), nullable=False)
    can_share = Column(Boolean, nullable=False, default=False)
    status = Column(String(20), nullable=False, default='pending', server_default='pending')