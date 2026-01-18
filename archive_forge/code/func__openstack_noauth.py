import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
def _openstack_noauth(self, cmd, *args, **kwargs):
    exe = os.path.join(cfg.CONF.designateclient.directory, 'openstack')
    options = build_option_string({'--os-url': cfg.CONF.identity.override_endpoint, '--os-token': cfg.CONF.identity.override_token})
    cmd = options + ' ' + cmd
    return base.execute(exe, cmd, *args, **kwargs)