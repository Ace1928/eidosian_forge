from openstack.tests.functional.shared_file_system import base
class LimitTest(base.BaseSharedFileSystemTest):

    def test_limits(self):
        limits = self.user_cloud.shared_file_system.limits()
        self.assertGreater(len(list(limits)), 0)
        for limit in limits:
            for attribute in ('maxTotalReplicaGigabytes', 'maxTotalShares', 'maxTotalShareGigabytes', 'maxTotalShareNetworks', 'maxTotalShareSnapshots', 'maxTotalShareReplicas', 'maxTotalSnapshotGigabytes', 'totalReplicaGigabytesUsed', 'totalShareGigabytesUsed', 'totalSharesUsed', 'totalShareNetworksUsed', 'totalShareSnapshotsUsed', 'totalSnapshotGigabytesUsed', 'totalShareReplicasUsed'):
                self.assertTrue(hasattr(limit, attribute))