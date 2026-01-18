import tempfile
from unittest import mock
import testtools
import openstack.cloud.openstackcloud as oc_oc
from openstack import exceptions
from openstack.object_store.v1 import _proxy
from openstack.object_store.v1 import container
from openstack.object_store.v1 import obj
from openstack.tests.unit import base
from openstack import utils
class BaseTestObject(base.TestCase):

    def setUp(self):
        super(BaseTestObject, self).setUp()
        self.container = self.getUniqueString()
        self.object = self.getUniqueString()
        self.endpoint = self.cloud.object_store.get_endpoint()
        self.container_endpoint = '{endpoint}/{container}'.format(endpoint=self.endpoint, container=self.container)
        self.object_endpoint = '{endpoint}/{object}'.format(endpoint=self.container_endpoint, object=self.object)

    def _compare_containers(self, exp, real):
        self.assertDictEqual(container.Container(**exp).to_dict(computed=False), real.to_dict(computed=False))

    def _compare_objects(self, exp, real):
        self.assertDictEqual(obj.Object(**exp).to_dict(computed=False), real.to_dict(computed=False))