from oslo_utils import uuidutils
from novaclient.tests.functional import base
class TestMigrationList(base.ClientTestBase):
    """Tests the "nova migration-list" command."""

    def _filter_migrations(self, version, migration_type, source_compute):
        """
        Filters migrations by --migration-type and --source-compute.

        :param version: The --os-compute-api-version to use.
        :param migration_type: The type of migrations to filter.
        :param source_compute: The source compute service hostname to filter.
        :return: output of the nova migration-list command with filters applied
        """
        return self.nova('migration-list', flags='--os-compute-api-version %s' % version, params='--migration-type %s --source-compute %s' % (migration_type, source_compute))

    def test_migration_list(self):
        """Tests creating a server, resizing it and then listing and filtering
        migrations using various microversion milestones.
        """
        server_id = self._create_server(flavor=self.flavor.id).id
        server = self.nova('show', params='%s' % server_id)
        server_user_id = self._get_value_from_the_table(server, 'user_id')
        tenant_id = self._get_value_from_the_table(server, 'tenant_id')
        source_compute = self._get_value_from_the_table(server, 'OS-EXT-SRV-ATTR:host')
        alternate_flavor = self._pick_alternate_flavor()
        self.nova('resize', params='%s %s --poll' % (server_id, alternate_flavor))
        self.nova('resize-confirm', params='%s' % server_id)
        self._wait_for_state_change(server_id, 'active')
        migrations = self.nova('migration-list', flags='--os-compute-api-version 2.1')
        instance_uuid = self._get_column_value_from_single_row_table(migrations, 'Instance UUID')
        self.assertEqual(server_id, instance_uuid)
        migration_status = self._get_column_value_from_single_row_table(migrations, 'Status')
        self.assertEqual('confirmed', migration_status)
        migrations = self.nova('migration-list', flags='--os-compute-api-version 2.23')
        migration_type = self._get_column_value_from_single_row_table(migrations, 'Type')
        self.assertEqual('resize', migration_type)
        migrations = self._filter_migrations('2.1', 'resize', source_compute)
        src_compute = self._get_column_value_from_single_row_table(migrations, 'Source Compute')
        self.assertEqual(source_compute, src_compute)
        migrations = self._filter_migrations('2.59', 'resize', source_compute)
        self._get_column_value_from_single_row_table(migrations, 'UUID')
        migrations = self._filter_migrations('2.66', 'resize', source_compute)
        self._get_column_value_from_single_row_table(migrations, 'UUID')
        migrations = self._filter_migrations('2.1', 'evacuation', source_compute)
        self.assertNotIn(server_id, migrations)
        migrations = self._filter_migrations('2.66', 'resize', uuidutils.generate_uuid())
        self.assertNotIn(server_id, migrations)
        migrations = self.nova('migration-list', flags='--os-compute-api-version 2.80')
        user_id = self._get_column_value_from_single_row_table(migrations, 'User ID')
        self.assertEqual(server_user_id, user_id)
        project_id = self._get_column_value_from_single_row_table(migrations, 'Project ID')
        self.assertEqual(tenant_id, project_id)