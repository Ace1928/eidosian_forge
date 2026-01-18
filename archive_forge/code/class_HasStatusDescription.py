from oslo_db.sqlalchemy import models
from oslo_utils import uuidutils
import sqlalchemy as sa
from sqlalchemy.ext import declarative
from sqlalchemy import orm
from neutron_lib.db import constants as db_const
class HasStatusDescription(object):
    """Status with description mixin."""
    status = sa.Column(sa.String(db_const.STATUS_FIELD_SIZE), nullable=False)
    status_description = sa.Column(sa.String(db_const.DESCRIPTION_FIELD_SIZE))