import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from osc_lib import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_port
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def _test_baremetal_port_create_llc_warning(self, additional_args, additional_verify_items):
    arglist = [baremetal_fakes.baremetal_port_address, '--node', baremetal_fakes.baremetal_uuid]
    arglist.extend(additional_args)
    verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('address', baremetal_fakes.baremetal_port_address)]
    verifylist.extend(additional_verify_items)
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.log = mock.Mock()
    self.cmd.take_action(parsed_args)
    args = {'address': baremetal_fakes.baremetal_port_address, 'node_uuid': baremetal_fakes.baremetal_uuid, 'local_link_connection': {'switch_id': 'aa:bb:cc:dd:ee:ff', 'port_id': 'eth0'}}
    self.baremetal_mock.port.create.assert_called_once_with(**args)
    self.cmd.log.warning.assert_called()