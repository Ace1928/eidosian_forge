import ctypes
import errno
import json
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_service import loopingcall
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.connectors import base_rbd
from os_brick.initiator.windows import base as win_conn_base
from os_brick import utils
def _wait_for_volume(self, connection_properties):
    """Wait for the specified volume to become accessible."""
    attempt = 0
    dev_path = None

    def _check_rbd_device():
        rbd_dev_path = self.get_device_name(connection_properties, expect=False)
        if rbd_dev_path:
            try:
                with open(rbd_dev_path, 'rb'):
                    pass
                nonlocal dev_path
                dev_path = rbd_dev_path
                raise loopingcall.LoopingCallDone()
            except FileNotFoundError:
                LOG.debug("The RBD image %(image)s mapped to local device %(dev)s isn't available yet.", {'image': connection_properties['name'], 'dev': rbd_dev_path})
        nonlocal attempt
        attempt += 1
        if attempt >= self.device_scan_attempts:
            msg = _("The mounted RBD image isn't available: %s")
            raise exception.VolumeDeviceNotFound(msg % connection_properties['name'])
    timer = loopingcall.FixedIntervalLoopingCall(_check_rbd_device)
    timer.start(interval=self.device_scan_interval).wait()
    return dev_path