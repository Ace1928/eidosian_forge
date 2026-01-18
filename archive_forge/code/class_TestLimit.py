from openstack.block_storage.v2 import limits
from openstack.tests.unit import base
class TestLimit(base.TestCase):

    def test_basic(self):
        limit_resource = limits.Limit()
        self.assertEqual('limits', limit_resource.resource_key)
        self.assertEqual('/limits', limit_resource.base_path)
        self.assertTrue(limit_resource.allow_fetch)
        self.assertFalse(limit_resource.allow_create)
        self.assertFalse(limit_resource.allow_commit)
        self.assertFalse(limit_resource.allow_delete)
        self.assertFalse(limit_resource.allow_list)

    def _test_absolute_limit(self, expected, actual):
        self.assertEqual(expected['totalSnapshotsUsed'], actual.total_snapshots_used)
        self.assertEqual(expected['maxTotalBackups'], actual.max_total_backups)
        self.assertEqual(expected['maxTotalVolumeGigabytes'], actual.max_total_volume_gigabytes)
        self.assertEqual(expected['maxTotalSnapshots'], actual.max_total_snapshots)
        self.assertEqual(expected['maxTotalBackupGigabytes'], actual.max_total_backup_gigabytes)
        self.assertEqual(expected['totalBackupGigabytesUsed'], actual.total_backup_gigabytes_used)
        self.assertEqual(expected['maxTotalVolumes'], actual.max_total_volumes)
        self.assertEqual(expected['totalVolumesUsed'], actual.total_volumes_used)
        self.assertEqual(expected['totalBackupsUsed'], actual.total_backups_used)
        self.assertEqual(expected['totalGigabytesUsed'], actual.total_gigabytes_used)

    def _test_rate_limit(self, expected, actual):
        self.assertEqual(expected[0]['verb'], actual[0].verb)
        self.assertEqual(expected[0]['value'], actual[0].value)
        self.assertEqual(expected[0]['remaining'], actual[0].remaining)
        self.assertEqual(expected[0]['unit'], actual[0].unit)
        self.assertEqual(expected[0]['next-available'], actual[0].next_available)

    def _test_rate_limits(self, expected, actual):
        self.assertEqual(expected[0]['regex'], actual[0].regex)
        self.assertEqual(expected[0]['uri'], actual[0].uri)
        self._test_rate_limit(expected[0]['limit'], actual[0].limits)

    def test_make_limit(self):
        limit_resource = limits.Limit(**LIMIT)
        self._test_rate_limits(LIMIT['rate'], limit_resource.rate)
        self._test_absolute_limit(LIMIT['absolute'], limit_resource.absolute)