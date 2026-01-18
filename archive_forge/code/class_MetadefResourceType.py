from oslo_db.sqlalchemy import models
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Index
from sqlalchemy import Integer
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import UniqueConstraint
from glance.common import timeutils
from glance.db.sqlalchemy.models import JSONEncodedDict
class MetadefResourceType(BASE_DICT, GlanceMetadefBase):
    """Represents a metadata-schema resource type in the datastore."""
    __tablename__ = 'metadef_resource_types'
    __table_args__ = (UniqueConstraint('name', name='uq_metadef_resource_types_name'),)
    id = Column(Integer, primary_key=True, nullable=False)
    name = Column(String(80), nullable=False)
    protected = Column(Boolean, nullable=False, default=False)
    associations = relationship('MetadefNamespaceResourceType', primaryjoin=id == MetadefNamespaceResourceType.resource_type_id)