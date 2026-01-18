from oslotest import base
from neutron_lib.db import resource_extend
from neutron_lib import fixture
class TestResourceExtend(base.BaseTestCase):

    def setUp(self):
        super(TestResourceExtend, self).setUp()
        self.useFixture(fixture.DBResourceExtendFixture())

    def test_register_funcs(self):
        resources = ['A', 'B', 'C']
        for r in resources:
            resource_extend.register_funcs(r, (lambda x: x,))
        for r in resources:
            self.assertIsNotNone(resource_extend.get_funcs(r))

    def test_apply_funcs(self):
        resources = ['A', 'B', 'C']
        callbacks = []

        def _cb(resp, db_obj):
            callbacks.append(resp)
        for r in resources:
            resource_extend.register_funcs(r, (_cb,))
        for r in resources:
            resource_extend.apply_funcs(r, None, None)
        self.assertEqual(3, len(callbacks))