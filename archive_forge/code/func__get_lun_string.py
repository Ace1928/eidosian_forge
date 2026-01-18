from oslo_log import log as logging
from os_brick import initiator
from os_brick.initiator.connectors import fibre_channel
from os_brick.initiator import linuxfc
def _get_lun_string(self, lun):
    target_lun = 0
    if lun <= 65535:
        target_lun = '0x%04x000000000000' % lun
    elif lun <= 4294967295:
        target_lun = '0x%08x00000000' % lun
    return target_lun