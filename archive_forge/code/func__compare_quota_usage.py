from novaclient.tests.functional import base
def _compare_quota_usage(self, old_usage, new_usage, expect_diff=True):
    """Compares the quota usage in the provided AbsoluteLimits."""
    self.assertEqual(old_usage['totalInstancesUsed'], new_usage['totalInstancesUsed'], 'totalInstancesUsed does not match')
    self.assertEqual(old_usage['totalCoresUsed'], new_usage['totalCoresUsed'], 'totalCoresUsed does not match')
    if expect_diff:
        self.assertNotEqual(old_usage['totalRAMUsed'], new_usage['totalRAMUsed'], 'totalRAMUsed should have changed')
    else:
        self.assertEqual(old_usage['totalRAMUsed'], new_usage['totalRAMUsed'], 'totalRAMUsed does not match')