import uuid
import sqlalchemy
from keystone.common import sql
from keystone.endpoint_policy.backends import base
from keystone import exception
class PolicyAssociation(sql.ModelBase, sql.ModelDictMixin):
    __tablename__ = 'policy_association'
    attributes = ['policy_id', 'endpoint_id', 'region_id', 'service_id']
    id = sql.Column(sql.String(64), primary_key=True)
    policy_id = sql.Column(sql.String(64), nullable=False)
    endpoint_id = sql.Column(sql.String(64), nullable=True)
    service_id = sql.Column(sql.String(64), nullable=True)
    region_id = sql.Column(sql.String(64), nullable=True)
    __table_args__ = (sql.UniqueConstraint('endpoint_id', 'service_id', 'region_id'),)

    def to_dict(self):
        """Return the model's attributes as a dictionary.

        We override the standard method in order to hide the id column,
        since this only exists to provide the table with a primary key.

        """
        d = {}
        for attr in self.__class__.attributes:
            d[attr] = getattr(self, attr)
        return d