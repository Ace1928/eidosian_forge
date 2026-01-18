import datetime
import sqlalchemy
from keystone.application_credential.backends import base
from keystone.common import password_hashing
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
class AccessRuleModel(sql.ModelBase, sql.ModelDictMixin):
    __tablename__ = 'access_rule'
    attributes = ['external_id', 'user_id', 'service', 'path', 'method']
    id = sql.Column(sql.Integer, primary_key=True, nullable=False)
    external_id = sql.Column(sql.String(64), index=True, unique=True)
    user_id = sql.Column(sql.String(64), index=True)
    service = sql.Column(sql.String(64))
    path = sql.Column(sql.String(128))
    method = sql.Column(sql.String(16))
    __table_args__ = (sql.UniqueConstraint('user_id', 'service', 'path', 'method', name='duplicate_access_rule_for_user_constraint'),)
    application_credential = sqlalchemy.orm.relationship('ApplicationCredentialAccessRuleModel', backref=sqlalchemy.orm.backref('access_rule'), cascade_backrefs=False)