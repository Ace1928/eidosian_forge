import copy
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import qos_specs
class TestQosAssociate(TestQos):
    volume_type = volume_fakes.create_one_volume_type()
    qos_spec = volume_fakes.create_one_qos()

    def setUp(self):
        super(TestQosAssociate, self).setUp()
        self.qos_mock.get.return_value = self.qos_spec
        self.types_mock.get.return_value = self.volume_type
        self.cmd = qos_specs.AssociateQos(self.app, None)

    def test_qos_associate(self):
        arglist = [self.qos_spec.id, self.volume_type.id]
        verifylist = [('qos_spec', self.qos_spec.id), ('volume_type', self.volume_type.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.qos_mock.associate.assert_called_with(self.qos_spec.id, self.volume_type.id)
        self.assertIsNone(result)