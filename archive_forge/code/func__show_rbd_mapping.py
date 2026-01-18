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
def _show_rbd_mapping(self, connection_properties):
    cmd = ['rbd-wnbd', 'show', connection_properties['name'], '--format', 'json']
    try:
        out, err = self._execute(*cmd)
        return json.loads(out)
    except processutils.ProcessExecutionError as ex:
        if abs(ctypes.c_int32(ex.exit_code).value) == errno.ENOENT:
            LOG.debug("Couldn't find RBD mapping: %s", connection_properties['name'])
            return
        raise
    except json.decoder.JSONDecodeError:
        msg = _('Could not get rbd mappping.')
        LOG.exception(msg)
        raise exception.BrickException(msg)