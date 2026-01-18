from unittest import mock
from cinderclient.v3 import client as cinderclient
from heat.engine.clients.os import cinder
from heat.engine.clients.os import nova
from heat.engine.resources.aws.ec2 import volume as aws_vol
from heat.engine.resources.openstack.cinder import volume as os_vol
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def _mock_create_server_volume_script(self, fva, final_status='in-use', update=False, extra_create_server_volume_mocks=[]):
    if not update:
        nova.NovaClientPlugin.client.return_value = self.fc
    result = [fva]
    for m in extra_create_server_volume_mocks:
        result.append(m)
    prev = self.fc.volumes.create_server_volume.side_effect or []
    self.fc.volumes.create_server_volume.side_effect = list(prev) + result
    fv_ready = FakeVolume(final_status, id=fva.id)
    return fv_ready