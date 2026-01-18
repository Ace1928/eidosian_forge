from unittest import mock
from mistralclient.auth import keystone
from heat.common import exception
from heat.engine.clients.os import mistral
from heat.tests import common
from heat.tests import utils
class MistralClientPluginTest(common.HeatTestCase):

    def test_create(self):
        self.patchobject(keystone.KeystoneAuthHandler, 'authenticate')
        context = utils.dummy_context()
        plugin = context.clients.client_plugin('mistral')
        client = plugin.client()
        self.assertIsNotNone(client.workflows)