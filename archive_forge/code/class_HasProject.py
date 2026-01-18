from oslo_db.sqlalchemy import models
from oslo_utils import uuidutils
import sqlalchemy as sa
from sqlalchemy.ext import declarative
from sqlalchemy import orm
from neutron_lib.db import constants as db_const
class HasProject(object):
    """Project mixin, add to subclasses that have a user."""
    project_id = sa.Column(sa.String(db_const.PROJECT_ID_FIELD_SIZE), index=True)

    def get_tenant_id(self):
        return self.project_id

    def set_tenant_id(self, value):
        self.project_id = value

    @declarative.declared_attr
    def tenant_id(cls):
        return orm.synonym('project_id', descriptor=property(cls.get_tenant_id, cls.set_tenant_id))