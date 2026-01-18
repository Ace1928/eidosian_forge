import collections
import time
from os_win import exceptions as os_win_exc
from os_win import utilsfactory
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.windows import base as win_conn_base
from os_brick import utils
def _get_disk_paths_by_scsi_id(self, connection_properties, fcp_lun):
    for local_port_wwn, remote_port_wwns in connection_properties['initiator_target_map'].items():
        for remote_port_wwn in remote_port_wwns:
            try:
                dev_nums = self._get_dev_nums_by_scsi_id(local_port_wwn, remote_port_wwn, fcp_lun)
                disk_paths = [self._diskutils.get_device_name_by_device_number(dev_num) for dev_num in dev_nums]
                return disk_paths
            except os_win_exc.FCException as ex:
                LOG.debug('Failed to retrieve volume paths by SCSI id. Exception: %s', ex)
                continue
    return []