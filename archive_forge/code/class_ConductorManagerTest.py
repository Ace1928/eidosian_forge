import testtools
from testtools.matchers import HasLength
from ironicclient.tests.unit import utils
from ironicclient.v1 import conductor
class ConductorManagerTest(testtools.TestCase):

    def setUp(self):
        super(ConductorManagerTest, self).setUp()
        self.api = utils.FakeAPI(fake_responses)
        self.mgr = conductor.ConductorManager(self.api)

    def test_conductor_list(self):
        conductors = self.mgr.list()
        expect = [('GET', '/v1/conductors', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(2, len(conductors))

    def test_conductor_list_detail(self):
        conductors = self.mgr.list(detail=True)
        expect = [('GET', '/v1/conductors/?detail=True', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(2, len(conductors))

    def test_conductor_list_limit(self):
        self.api = utils.FakeAPI(fake_responses_pagination)
        self.mgr = conductor.ConductorManager(self.api)
        conductors = self.mgr.list(limit=1)
        expect = [('GET', '/v1/conductors/?limit=1', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertThat(conductors, HasLength(1))

    def test_conductor_list_marker(self):
        self.api = utils.FakeAPI(fake_responses_pagination)
        self.mgr = conductor.ConductorManager(self.api)
        conductors = self.mgr.list(marker=CONDUCTOR1['hostname'])
        expect = [('GET', '/v1/conductors/?marker=%s' % CONDUCTOR1['hostname'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertThat(conductors, HasLength(1))

    def test_conductor_list_pagination_no_limit(self):
        self.api = utils.FakeAPI(fake_responses_pagination)
        self.mgr = conductor.ConductorManager(self.api)
        conductors = self.mgr.list(limit=0)
        expect = [('GET', '/v1/conductors', {}, None), ('GET', '/v1/conductors/?limit=1', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(2, len(conductors))

    def test_conductor_list_sort_key(self):
        self.api = utils.FakeAPI(fake_responses_sorting)
        self.mgr = conductor.ConductorManager(self.api)
        conductors = self.mgr.list(sort_key='updated_at')
        expect = [('GET', '/v1/conductors/?sort_key=updated_at', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(2, len(conductors))

    def test_conductor_list_sort_dir(self):
        self.api = utils.FakeAPI(fake_responses_sorting)
        self.mgr = conductor.ConductorManager(self.api)
        conductors = self.mgr.list(sort_dir='desc')
        expect = [('GET', '/v1/conductors/?sort_dir=desc', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(2, len(conductors))

    def test_conductor_list_fields(self):
        conductors = self.mgr.list(fields=['hostname', 'alive'])
        expect = [('GET', '/v1/conductors/?fields=hostname,alive', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(conductors))

    def test_conductor_show(self):
        conductor = self.mgr.get(CONDUCTOR1['hostname'])
        expect = [('GET', '/v1/conductors/%s' % CONDUCTOR1['hostname'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(CONDUCTOR1['hostname'], conductor.hostname)

    def test_conductor_show_fields(self):
        conductor = self.mgr.get(CONDUCTOR1['hostname'], fields=['hostname', 'alive'])
        expect = [('GET', '/v1/conductors/%s?fields=hostname,alive' % CONDUCTOR1['hostname'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(CONDUCTOR1['hostname'], conductor.hostname)