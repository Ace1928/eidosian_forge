from oslo_log import log
from oslo_serialization import jsonutils
from sqlalchemy import orm
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.federation.backends import base
from keystone.i18n import _
class MappingModel(sql.ModelBase, sql.ModelDictMixin):
    __tablename__ = 'mapping'
    attributes = ['id', 'rules', 'schema_version']
    id = sql.Column(sql.String(64), primary_key=True)
    rules = sql.Column(sql.JsonBlob(), nullable=False)
    schema_version = sql.Column(sql.String(5), nullable=False, server_default='1.0')

    @classmethod
    def from_dict(cls, dictionary):
        new_dictionary = dictionary.copy()
        new_dictionary['rules'] = jsonutils.dumps(new_dictionary['rules'])
        return cls(**new_dictionary)

    def to_dict(self):
        """Return a dictionary with model's attributes."""
        d = dict()
        for attr in self.__class__.attributes:
            d[attr] = getattr(self, attr)
        d['rules'] = jsonutils.loads(d['rules'])
        return d