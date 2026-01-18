import datetime
import sqlalchemy
from keystone.application_credential.backends import base
from keystone.common import password_hashing
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
def _check_secret(self, secret, app_cred_ref):
    secret_hash = app_cred_ref['secret_hash']
    return password_hashing.check_password(secret, secret_hash)