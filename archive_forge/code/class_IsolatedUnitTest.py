import os
from unittest import mock
import glance_store as store
from glance_store._drivers import cinder
from glance_store._drivers import rbd as rbd_store
from glance_store._drivers import swift
from glance_store import location
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_db import options
from oslo_serialization import jsonutils
from glance.tests import stubs
from glance.tests import utils as test_utils
class IsolatedUnitTest(StoreClearingUnitTest):
    """
    Unit test case that establishes a mock environment within
    a testing directory (in isolation)
    """

    def setUp(self):
        super(IsolatedUnitTest, self).setUp()
        options.set_defaults(CONF, connection='sqlite://')
        lockutils.set_defaults(os.path.join(self.test_dir))
        self.config(debug=False)
        self.config(default_store='filesystem', filesystem_store_datadir=self.test_dir, group='glance_store')
        store.create_stores()

        def fake_get_conection_type(client):
            DEFAULT_API_PORT = 9292
            if client.port == DEFAULT_API_PORT:
                return stubs.FakeGlanceConnection
        self.patcher = mock.patch('glance.common.client.BaseClient.get_connection_type', fake_get_conection_type)
        self.addCleanup(self.patcher.stop)
        self.patcher.start()

    def set_policy_rules(self, rules):
        fap = open(CONF.oslo_policy.policy_file, 'w')
        fap.write(jsonutils.dumps(rules))
        fap.close()