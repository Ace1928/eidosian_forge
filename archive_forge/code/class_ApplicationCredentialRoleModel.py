import datetime
import sqlalchemy
from keystone.application_credential.backends import base
from keystone.common import password_hashing
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
class ApplicationCredentialRoleModel(sql.ModelBase, sql.ModelDictMixin):
    __tablename__ = 'application_credential_role'
    attributes = ['application_credential_id', 'role_id']
    application_credential_id = sql.Column(sql.Integer, sql.ForeignKey('application_credential.internal_id', ondelete='cascade'), primary_key=True, nullable=False)
    role_id = sql.Column(sql.String(64), primary_key=True, nullable=False)