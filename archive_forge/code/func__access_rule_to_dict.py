import datetime
import sqlalchemy
from keystone.application_credential.backends import base
from keystone.common import password_hashing
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
def _access_rule_to_dict(self, ref):
    access_rule = ref.to_dict()
    return {k.replace('external_id', 'id'): v for k, v in access_rule.items() if k != 'user_id' and k != 'id'}