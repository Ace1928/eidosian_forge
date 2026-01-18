import datetime
import sqlalchemy
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import orm
from sqlalchemy.orm import collections
from keystone.common import password_hashing
from keystone.common import resource_options
from keystone.common import sql
import keystone.conf
from keystone.identity.backends import resource_options as iro
class NonLocalUser(sql.ModelBase, sql.ModelDictMixin):
    """SQL data model for nonlocal users (LDAP and custom)."""
    __tablename__ = 'nonlocal_user'
    attributes = ['domain_id', 'name', 'user_id']
    domain_id = sql.Column(sql.String(64), primary_key=True)
    name = sql.Column(sql.String(255), primary_key=True)
    user_id = sql.Column(sql.String(64), nullable=False)
    __table_args__ = (sql.UniqueConstraint('user_id'), sqlalchemy.ForeignKeyConstraint(['user_id', 'domain_id'], ['user.id', 'user.domain_id'], onupdate='CASCADE', ondelete='CASCADE'))