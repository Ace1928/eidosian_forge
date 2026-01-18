from __future__ import annotations
from typing import Any
from typing import Dict
from typing import Set
from sqlalchemy import CHAR
from sqlalchemy import CheckConstraint
from sqlalchemy import Column
from sqlalchemy import event
from sqlalchemy import ForeignKey
from sqlalchemy import Index
from sqlalchemy import inspect
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import Numeric
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import Text
from sqlalchemy import text
from sqlalchemy import UniqueConstraint
from ... import autogenerate
from ... import util
from ...autogenerate import api
from ...ddl.base import _fk_spec
from ...migration import MigrationContext
from ...operations import ops
from ...testing import config
from ...testing import eq_
from ...testing.env import clear_staging_env
from ...testing.env import staging_env
class AutogenFixtureTest(_ComparesFKs):

    def _fixture(self, m1, m2, include_schemas=False, opts=None, object_filters=_default_object_filters, name_filters=_default_name_filters, return_ops=False, max_identifier_length=None):
        if max_identifier_length:
            dialect = self.bind.dialect
            existing_length = dialect.max_identifier_length
            dialect.max_identifier_length = dialect._user_defined_max_identifier_length = max_identifier_length
        try:
            self._alembic_metadata, model_metadata = (m1, m2)
            for m in util.to_list(self._alembic_metadata):
                m.create_all(self.bind)
            with self.bind.connect() as conn:
                ctx_opts = {'compare_type': True, 'compare_server_default': True, 'target_metadata': model_metadata, 'upgrade_token': 'upgrades', 'downgrade_token': 'downgrades', 'alembic_module_prefix': 'op.', 'sqlalchemy_module_prefix': 'sa.', 'include_object': object_filters, 'include_name': name_filters, 'include_schemas': include_schemas}
                if opts:
                    ctx_opts.update(opts)
                self.context = context = MigrationContext.configure(connection=conn, opts=ctx_opts)
                autogen_context = api.AutogenContext(context, model_metadata)
                uo = ops.UpgradeOps(ops=[])
                autogenerate._produce_net_changes(autogen_context, uo)
                if return_ops:
                    return uo
                else:
                    return uo.as_diffs()
        finally:
            if max_identifier_length:
                dialect = self.bind.dialect
                dialect.max_identifier_length = dialect._user_defined_max_identifier_length = existing_length

    def setUp(self):
        staging_env()
        self.bind = config.db

    def tearDown(self):
        if hasattr(self, '_alembic_metadata'):
            for m in util.to_list(self._alembic_metadata):
                m.drop_all(self.bind)
        clear_staging_env()