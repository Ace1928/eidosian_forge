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
class MetadefNamespace(BASE_DICT, GlanceMetadefBase):
    """Represents a metadata-schema namespace in the datastore."""
    __tablename__ = 'metadef_namespaces'
    __table_args__ = (UniqueConstraint('namespace', name='uq_metadef_namespaces_namespace'), Index('ix_metadef_namespaces_owner', 'owner'))
    id = Column(Integer, primary_key=True, nullable=False)
    namespace = Column(String(80), nullable=False)
    display_name = Column(String(80))
    description = Column(Text())
    visibility = Column(String(32))
    protected = Column(Boolean)
    owner = Column(String(255), nullable=False)