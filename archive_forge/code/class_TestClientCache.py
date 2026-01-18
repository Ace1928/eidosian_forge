import copy
from unittest import mock
from keystoneauth1.access import service_catalog
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1.identity import generic as generic_plugin
from keystoneauth1.identity.v3 import k2k
from keystoneauth1 import loading
from keystoneauth1 import noauth
from keystoneauth1 import token_endpoint
from openstack.config import cloud_config
from openstack.config import defaults
from openstack import connection
from osc_lib.api import auth
from osc_lib import clientmanager
from osc_lib import exceptions as exc
from osc_lib.tests import fakes
from osc_lib.tests import utils
class TestClientCache(utils.TestCase):

    def test_singleton(self):
        c = Container()
        self.assertEqual(c.attr, c.attr)

    def test_attribute_error_propagates(self):
        c = Container()
        err = self.assertRaises(exc.PluginAttributeError, getattr, c, 'buggy_attr')
        self.assertNotIsInstance(err, AttributeError)
        self.assertEqual("'Container' object has no attribute 'foo'", str(err))