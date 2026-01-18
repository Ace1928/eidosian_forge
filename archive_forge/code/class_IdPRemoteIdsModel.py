from oslo_log import log
from oslo_serialization import jsonutils
from sqlalchemy import orm
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.federation.backends import base
from keystone.i18n import _
class IdPRemoteIdsModel(sql.ModelBase, sql.ModelDictMixin):
    __tablename__ = 'idp_remote_ids'
    attributes = ['idp_id', 'remote_id']
    mutable_attributes = frozenset(['idp_id', 'remote_id'])
    idp_id = sql.Column(sql.String(64), sql.ForeignKey('identity_provider.id', ondelete='CASCADE'))
    remote_id = sql.Column(sql.String(255), primary_key=True)

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