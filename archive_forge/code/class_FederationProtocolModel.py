from oslo_log import log
from oslo_serialization import jsonutils
from sqlalchemy import orm
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.federation.backends import base
from keystone.i18n import _
class FederationProtocolModel(sql.ModelBase, sql.ModelDictMixin):
    __tablename__ = 'federation_protocol'
    attributes = ['id', 'idp_id', 'mapping_id', 'remote_id_attribute']
    mutable_attributes = frozenset(['mapping_id', 'remote_id_attribute'])
    id = sql.Column(sql.String(64), primary_key=True)
    idp_id = sql.Column(sql.String(64), sql.ForeignKey('identity_provider.id', ondelete='CASCADE'), primary_key=True)
    mapping_id = sql.Column(sql.String(64), nullable=False)
    remote_id_attribute = sql.Column(sql.String(64))

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