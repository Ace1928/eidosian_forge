from __future__ import annotations
from collections import defaultdict
import copy
import glob
import os
import re
import time
from typing import Any, Iterable, Optional, Union  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import strutils
from os_brick import exception
from os_brick import executor
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import base_iscsi
from os_brick.initiator import utils as initiator_utils
from os_brick import utils
@utils.retry(exception.VolumeDeviceNotFound)
def _connect_multipath_volume(self, connection_properties: dict) -> Optional[dict[str, str]]:
    """Connect to a multipathed volume launching parallel login requests.

        We will be doing parallel login requests, which will considerably speed
        up the process when we have flaky connections.

        We'll always try to return a multipath device even if there's only one
        path discovered, that way we can return once we have logged in in all
        the portals, because the paths will come up later.

        To make this possible we tell multipathd that the wwid is a multipath
        as soon as we have one device, and then hint multipathd to reconsider
        that volume for a multipath asking to add the path, because even if
        it's already known by multipathd it would have been discarded if it
        was the first time this volume was seen here.
        """
    wwn: Optional[str] = None
    mpath = None
    wwn_added = False
    last_try_on = 0.0
    found: list = []
    just_added_devices: list = []
    data = {'stop_connecting': False, 'num_logins': 0, 'failed_logins': 0, 'stopped_threads': 0, 'found_devices': found, 'just_added_devices': just_added_devices}
    ips_iqns_luns = self._get_ips_iqns_luns(connection_properties)
    retries = self.device_scan_attempts
    threads = []
    for ip, iqn, lun in ips_iqns_luns:
        props = connection_properties.copy()
        props.update(target_portal=ip, target_iqn=iqn, target_lun=lun)
        for key in ('target_portals', 'target_iqns', 'target_luns'):
            props.pop(key, None)
        threads.append(executor.Thread(target=self._connect_vol, args=(retries, props, data)))
    for thread in threads:
        thread.start()
    while not (len(ips_iqns_luns) == data['stopped_threads'] and (not found) or (mpath and len(ips_iqns_luns) == data['num_logins'] + data['failed_logins'])):
        if not wwn and found:
            wwn = self._linuxscsi.get_sysfs_wwn(found, mpath)
        if not mpath and found:
            mpath = self._linuxscsi.find_sysfs_multipath_dm(found)
            if wwn and (not (mpath or wwn_added)):
                wwn_added = self._linuxscsi.multipath_add_wwid(wwn)
                while not mpath and just_added_devices:
                    device_path = '/dev/' + just_added_devices.pop(0)
                    self._linuxscsi.multipath_add_path(device_path)
                    mpath = self._linuxscsi.find_sysfs_multipath_dm(found)
        if not last_try_on and found and (len(ips_iqns_luns) == data['stopped_threads']):
            LOG.debug('All connection threads finished, giving 10 seconds for dm to appear.')
            last_try_on = time.time() + 10
        elif last_try_on and last_try_on < time.time():
            break
        time.sleep(1)
    data['stop_connecting'] = True
    for thread in threads:
        thread.join()
    if not found:
        raise exception.VolumeDeviceNotFound(device='')
    if not mpath:
        LOG.warning('No dm was created, connection to volume is probably bad and will perform poorly.')
    elif not wwn:
        wwn = self._linuxscsi.get_sysfs_wwn(found, mpath)
    assert wwn is not None
    return self._get_connect_result(connection_properties, wwn, found, mpath)