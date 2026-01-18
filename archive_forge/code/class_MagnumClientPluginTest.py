from unittest import mock
from magnumclient import exceptions as mc_exc
from heat.engine.clients.os import magnum as mc
from heat.tests import common
from heat.tests import utils
class MagnumClientPluginTest(common.HeatTestCase):

    def test_create(self):
        context = utils.dummy_context()
        plugin = context.clients.client_plugin('magnum')
        client = plugin.client()
        self.assertEqual('http://server.test:5000/v3', client.cluster_templates.api.session.auth.endpoint)