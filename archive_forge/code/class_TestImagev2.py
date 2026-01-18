import random
from unittest import mock
import uuid
from openstack.image.v2 import _proxy
from openstack.image.v2 import cache
from openstack.image.v2 import image
from openstack.image.v2 import member
from openstack.image.v2 import metadef_namespace
from openstack.image.v2 import metadef_object
from openstack.image.v2 import metadef_property
from openstack.image.v2 import metadef_resource_type
from openstack.image.v2 import service_info as _service_info
from openstack.image.v2 import task
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils
class TestImagev2(FakeClientMixin, utils.TestCommand):

    def setUp(self):
        super().setUp()
        self.app.client_manager.identity = identity_fakes.FakeIdentityv3Client(endpoint=fakes.AUTH_URL, token=fakes.AUTH_TOKEN)