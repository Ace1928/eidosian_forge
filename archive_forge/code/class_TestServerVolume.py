import collections
import copy
import getpass
import json
import tempfile
from unittest import mock
from unittest.mock import call
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
class TestServerVolume(TestServer):

    def setUp(self):
        super(TestServerVolume, self).setUp()
        self.methods = {'create_volume_attachment': None}
        self.servers = self.setup_sdk_servers_mock(count=1)
        self.volumes = self.setup_sdk_volumes_mock(count=1)
        attrs = {'server_id': self.servers[0].id, 'volume_id': self.volumes[0].id}
        self.volume_attachment = compute_fakes.create_one_volume_attachment(attrs=attrs)
        self.compute_sdk_client.create_volume_attachment.return_value = self.volume_attachment