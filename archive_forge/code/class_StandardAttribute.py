from oslo_utils import timeutils
import sqlalchemy as sa
from sqlalchemy import event  # noqa
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext import declarative
from sqlalchemy.orm import attributes
from sqlalchemy.orm import session as se
from neutron_lib._i18n import _
from neutron_lib.db import constants as db_const
from neutron_lib.db import model_base
from neutron_lib.db import sqlalchemytypes
class StandardAttribute(model_base.BASEV2):
    """Common table to associate all Neutron API resources.

    By having Neutron objects related to this table, we can associate new
    tables that apply to many Neutron objects (e.g. timestamps, rbac entries)
    to this table to avoid schema duplication while maintaining referential
    integrity.

    NOTE(kevinbenton): This table should not have more columns added to it
    unless we are absolutely certain the new column will have a value for
    every single type of Neutron resource. Otherwise this table will be filled
    with NULL entries for combinations that don't make sense. Additionally,
    by keeping this table small we can ensure that performance isn't adversely
    impacted for queries on objects.
    """
    id = sa.Column(sa.BigInteger().with_variant(sa.Integer(), 'sqlite'), primary_key=True, autoincrement=True)
    resource_type = sa.Column(sa.String(255), nullable=False)
    description = sa.Column(sa.String(db_const.DESCRIPTION_FIELD_SIZE))
    revision_number = sa.Column(sa.BigInteger().with_variant(sa.Integer(), 'sqlite'), server_default='0', nullable=False)
    created_at = sa.Column(sqlalchemytypes.TruncatedDateTime, default=timeutils.utcnow)
    updated_at = sa.Column(sqlalchemytypes.TruncatedDateTime, onupdate=timeutils.utcnow)
    __mapper_args__ = {'version_id_col': revision_number, 'confirm_deleted_rows': False, 'version_id_generator': False}

    def bump_revision(self):
        if self.revision_number is None:
            return
        self.revision_number += 1

    def _set_updated_revision_number(self, revision_number, updated_at):
        attributes.set_committed_value(self, 'revision_number', revision_number)
        attributes.set_committed_value(self, 'updated_at', updated_at)

    @property
    def _effective_standard_attribute_id(self):
        return self.id