import warnings
from os_win import utilsfactory
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator import initiator_connector
from os_brick import utils
def _check_device_paths(self, device_paths):
    if len(device_paths) > 1:
        err_msg = _('Multiple volume paths were found: %s. This can occur if multipath is used and MPIO is not properly configured, thus not claiming the device paths. This issue must be addressed urgently as it can lead to data corruption.')
        raise exception.BrickException(err_msg % device_paths)