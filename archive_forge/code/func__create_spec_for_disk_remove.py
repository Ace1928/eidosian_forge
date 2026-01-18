import os
import tempfile
from oslo_log import log as logging
from oslo_utils import fileutils
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator import initiator_connector
def _create_spec_for_disk_remove(self, session, disk_device):
    cf = session.vim.client.factory
    disk_spec = cf.create('ns0:VirtualDeviceConfigSpec')
    disk_spec.operation = 'remove'
    disk_spec.fileOperation = 'destroy'
    disk_spec.device = disk_device
    return disk_spec