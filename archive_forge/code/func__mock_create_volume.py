import collections
import copy
import json
from unittest import mock
from cinderclient import exceptions as cinder_exp
from novaclient import exceptions as nova_exp
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine.resources.openstack.cinder import volume as c_vol
from heat.engine.resources import scheduler_hints as sh
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.objects import resource_data as resource_data_object
from heat.tests.openstack.cinder import test_volume_utils as vt_base
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def _mock_create_volume(self, fv, stack_name, size=1, final_status='available', extra_get_mocks=[], extra_create_mocks=[]):
    result = [fv]
    for m in extra_create_mocks:
        result.append(m)
    self.cinder_fc.volumes.create.side_effect = result
    fv_ready = vt_base.FakeVolume(final_status, id=fv.id)
    result = [fv, fv_ready]
    for m in extra_get_mocks:
        result.append(m)
    self.cinder_fc.volumes.get.side_effect = result
    return fv_ready