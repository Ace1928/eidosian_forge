from unittest import mock
import sqlalchemy as sa
from sqlalchemy import orm
from oslo_db.sqlalchemy import test_migrations as migrate
from oslo_db.tests.sqlalchemy import base as db_test_base
def _test_models_not_sync_filtered(self):
    self.metadata_migrations.clear()
    sa.Table('table', self.metadata_migrations, sa.Column('fk_check', sa.String(36), nullable=False), sa.PrimaryKeyConstraint('fk_check'), mysql_engine='InnoDB')
    sa.Table('testtbl', self.metadata_migrations, sa.Column('id', sa.Integer, primary_key=True), sa.Column('spam', sa.String(8), nullable=True), sa.Column('eggs', sa.DateTime), sa.Column('foo', sa.Boolean, server_default=sa.sql.expression.false()), sa.Column('bool_wo_default', sa.Boolean, unique=True), sa.Column('bar', sa.BigInteger), sa.Column('defaulttest', sa.Integer, server_default='7'), sa.Column('defaulttest2', sa.String(8), server_default=''), sa.Column('defaulttest3', sa.String(5), server_default='fake'), sa.Column('defaulttest4', sa.Enum('first', 'second', name='testenum'), server_default='first'), sa.Column('defaulttest5', sa.Integer, server_default=sa.text('0')), sa.Column('fk_check', sa.String(36), nullable=False), sa.UniqueConstraint('spam', 'foo', name='uniq_cons'), sa.ForeignKeyConstraint(['fk_check'], ['table.fk_check']), mysql_engine='InnoDB')
    with mock.patch.object(self, 'filter_metadata_diff') as filter_mock:

        def filter_diffs(diffs):
            return [diff for diff in diffs if 'constraint' in diff[0]]
        filter_mock.side_effect = filter_diffs
        msg = str(self.assertRaises(AssertionError, self.test_models_sync))
        self.assertNotIn('defaulttest', msg)
        self.assertNotIn('defaulttest3', msg)
        self.assertNotIn('remove_fk', msg)
        self.assertIn('constraint', msg)