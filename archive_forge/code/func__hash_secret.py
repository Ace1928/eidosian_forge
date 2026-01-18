import datetime
import sqlalchemy
from keystone.application_credential.backends import base
from keystone.common import password_hashing
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
def _hash_secret(self, app_cred_ref):
    unhashed_secret = app_cred_ref.pop('secret')
    hashed_secret = password_hashing.hash_password(unhashed_secret)
    app_cred_ref['secret_hash'] = hashed_secret