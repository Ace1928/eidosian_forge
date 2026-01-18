import sqlalchemy
from keystone.common import sql
from keystone.models import revoke_model
from keystone.revoke.backends import base
from oslo_db import api as oslo_db_api
def _flush_batch_size(self, dialect):
    batch_size = 0
    if dialect == 'ibm_db_sa':
        batch_size = 100
    return batch_size