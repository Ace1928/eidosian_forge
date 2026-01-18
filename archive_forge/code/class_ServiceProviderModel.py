from oslo_log import log
from oslo_serialization import jsonutils
from sqlalchemy import orm
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.federation.backends import base
from keystone.i18n import _
class ServiceProviderModel(sql.ModelBase, sql.ModelDictMixin):
    __tablename__ = 'service_provider'
    attributes = ['auth_url', 'id', 'enabled', 'description', 'relay_state_prefix', 'sp_url']
    mutable_attributes = frozenset(['auth_url', 'description', 'enabled', 'relay_state_prefix', 'sp_url'])
    id = sql.Column(sql.String(64), primary_key=True)
    enabled = sql.Column(sql.Boolean, nullable=False)
    description = sql.Column(sql.Text(), nullable=True)
    auth_url = sql.Column(sql.String(256), nullable=False)
    sp_url = sql.Column(sql.String(256), nullable=False)
    relay_state_prefix = sql.Column(sql.String(256), nullable=False)

    @classmethod
    def from_dict(cls, dictionary):
        new_dictionary = dictionary.copy()
        return cls(**new_dictionary)

    def to_dict(self):
        """Return a dictionary with model's attributes."""
        d = dict()
        for attr in self.__class__.attributes:
            d[attr] = getattr(self, attr)
        return d