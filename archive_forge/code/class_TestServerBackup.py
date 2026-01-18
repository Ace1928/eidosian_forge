from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server_backup
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
class TestServerBackup(compute_fakes.TestComputev2):

    def setup_servers_mock(self, count):
        servers = compute_fakes.create_sdk_servers(count=count)
        self.compute_sdk_client.find_server = compute_fakes.get_servers(servers, 0)
        return servers