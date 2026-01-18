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
class UserGroupMembership(sql.ModelBase, sql.ModelDictMixin):
    """Group membership join table."""
    __tablename__ = 'user_group_membership'
    user_id = sql.Column(sql.String(64), sql.ForeignKey('user.id'), primary_key=True)
    group_id = sql.Column(sql.String(64), sql.ForeignKey('group.id'), primary_key=True)