from novaclient.tests.functional import base
class TestAggregatesNovaClient(base.ClientTestBase):
    COMPUTE_API_VERSION = '2.1'

    def setUp(self):
        super(TestAggregatesNovaClient, self).setUp()
        self.agg1 = self.name_generate()
        self.agg2 = self.name_generate()
        self.addCleanup(self._clean_aggregates)

    def _clean_aggregates(self):
        for a in (self.agg1, self.agg2):
            try:
                self.nova('aggregate-delete', params=a)
            except Exception:
                pass

    def test_aggregate_update_name(self):
        self.nova('aggregate-create', params=self.agg1)
        self.nova('aggregate-update', params='--name=%s %s' % (self.agg2, self.agg1))
        output = self.nova('aggregate-show', params=self.agg2)
        self.assertIn(self.agg2, output)
        self.nova('aggregate-delete', params=self.agg2)

    def test_aggregate_update_az(self):
        self.nova('aggregate-create', params=self.agg2)
        self.nova('aggregate-update', params='--availability-zone=myaz %s' % self.agg2)
        output = self.nova('aggregate-show', params=self.agg2)
        self.assertIn('myaz', output)
        self.nova('aggregate-delete', params=self.agg2)