import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_target as bm_vol_target
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def _test_baremetal_volume_target_missing_param(self, missing):
    argdict = {'--node': baremetal_fakes.baremetal_uuid, '--type': baremetal_fakes.baremetal_volume_target_volume_type, '--boot-index': baremetal_fakes.baremetal_volume_target_boot_index, '--volume-id': baremetal_fakes.baremetal_volume_target_volume_id, '--uuid': baremetal_fakes.baremetal_volume_target_uuid}
    arglist = []
    for k, v in argdict.items():
        if k not in missing:
            arglist += [k, v]
    verifylist = None
    self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)