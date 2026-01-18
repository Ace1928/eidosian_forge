import copy
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import qos_specs
class TestQos(volume_fakes.TestVolume):

    def setUp(self):
        super(TestQos, self).setUp()
        self.qos_mock = self.volume_client.qos_specs
        self.qos_mock.reset_mock()
        self.types_mock = self.volume_client.volume_types
        self.types_mock.reset_mock()