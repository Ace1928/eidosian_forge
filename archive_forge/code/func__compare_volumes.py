import testtools
from openstack.block_storage.v3 import volume
from openstack.cloud import meta
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def _compare_volumes(self, exp, real):
    self.assertDictEqual(volume.Volume(**exp).to_dict(computed=False), real.to_dict(computed=False))