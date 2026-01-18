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
class TestPowerStateColumn(test_utils.TestCase):

    def test_human_readable(self):
        self.assertEqual('NOSTATE', server.PowerStateColumn(0).human_readable())
        self.assertEqual('Running', server.PowerStateColumn(1).human_readable())
        self.assertEqual('', server.PowerStateColumn(2).human_readable())
        self.assertEqual('Paused', server.PowerStateColumn(3).human_readable())
        self.assertEqual('Shutdown', server.PowerStateColumn(4).human_readable())
        self.assertEqual('', server.PowerStateColumn(5).human_readable())
        self.assertEqual('Crashed', server.PowerStateColumn(6).human_readable())
        self.assertEqual('Suspended', server.PowerStateColumn(7).human_readable())
        self.assertEqual('N/A', server.PowerStateColumn(8).human_readable())