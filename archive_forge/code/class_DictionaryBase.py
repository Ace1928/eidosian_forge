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
class DictionaryBase(models.ModelBase):
    metadata = None

    def to_dict(self):
        d = {}
        for c in self.__table__.columns:
            d[c.name] = self[c.name]
        return d