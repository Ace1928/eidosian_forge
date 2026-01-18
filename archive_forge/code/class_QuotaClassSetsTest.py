from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
class QuotaClassSetsTest(utils.TestCase):

    def test_class_quotas_get(self):
        class_name = 'test'
        cls = cs.quota_classes.get(class_name)
        cs.assert_called('GET', '/os-quota-class-sets/%s' % class_name)
        self._assert_request_id(cls)

    def test_update_quota(self):
        q = cs.quota_classes.get('test')
        q.update(volumes=2, snapshots=2, gigabytes=2000, backups=2, backup_gigabytes=2000, per_volume_gigabytes=100)
        cs.assert_called('PUT', '/os-quota-class-sets/test')
        self._assert_request_id(q)

    def test_refresh_quota(self):
        q = cs.quota_classes.get('test')
        q2 = cs.quota_classes.get('test')
        self.assertEqual(q.volumes, q2.volumes)
        self.assertEqual(q.snapshots, q2.snapshots)
        self.assertEqual(q.gigabytes, q2.gigabytes)
        self.assertEqual(q.backups, q2.backups)
        self.assertEqual(q.backup_gigabytes, q2.backup_gigabytes)
        self.assertEqual(q.per_volume_gigabytes, q2.per_volume_gigabytes)
        q2.volumes = 0
        self.assertNotEqual(q.volumes, q2.volumes)
        q2.snapshots = 0
        self.assertNotEqual(q.snapshots, q2.snapshots)
        q2.gigabytes = 0
        self.assertNotEqual(q.gigabytes, q2.gigabytes)
        q2.backups = 0
        self.assertNotEqual(q.backups, q2.backups)
        q2.backup_gigabytes = 0
        self.assertNotEqual(q.backup_gigabytes, q2.backup_gigabytes)
        q2.per_volume_gigabytes = 0
        self.assertNotEqual(q.per_volume_gigabytes, q2.per_volume_gigabytes)
        q2.get()
        self.assertEqual(q.volumes, q2.volumes)
        self.assertEqual(q.snapshots, q2.snapshots)
        self.assertEqual(q.gigabytes, q2.gigabytes)
        self.assertEqual(q.backups, q2.backups)
        self.assertEqual(q.backup_gigabytes, q2.backup_gigabytes)
        self.assertEqual(q.per_volume_gigabytes, q2.per_volume_gigabytes)
        self._assert_request_id(q)
        self._assert_request_id(q2)