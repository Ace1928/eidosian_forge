from oslo_utils import timeutils
import sqlalchemy
from sqlalchemy.ext.hybrid import hybrid_property
from keystone.common import sql
from keystone import exception
from keystone.trust.backends import base
class TrustRole(sql.ModelBase):
    __tablename__ = 'trust_role'
    attributes = ['trust_id', 'role_id']
    trust_id = sql.Column(sql.String(64), primary_key=True, nullable=False)
    role_id = sql.Column(sql.String(64), primary_key=True, nullable=False)