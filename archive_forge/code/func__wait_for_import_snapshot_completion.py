import re
import copy
import time
import base64
import warnings
from typing import List
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, basestring, ensure_string
from libcloud.utils.xml import findall, findattr, findtext, fixxpath
from libcloud.common.aws import DEFAULT_SIGNATURE_VERSION, AWSBaseResponse, SignedAWSConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date, parse_date_allow_empty
from libcloud.utils.publickey import get_pubkey_comment, get_pubkey_ssh2_fingerprint
from libcloud.compute.providers import Provider
from libcloud.compute.constants.ec2_region_details_partial import (
def _wait_for_import_snapshot_completion(self, import_task_id, timeout=1800, interval=15):
    """
        It waits for import snapshot to be completed

        :param import_task_id: Import task Id for the
                               current Import Snapshot Task
        :type import_task_id: ``str``

        :param timeout: Timeout value for snapshot generation
        :type timeout: ``float``

        :param interval: Time interval for repetitive describe
                         import snapshot tasks requests
        :type interval: ``float``

        :rtype: :class:``VolumeSnapshot``
        """
    start_time = time.time()
    snapshotId = None
    while snapshotId is None:
        if time.time() - start_time >= timeout:
            raise Exception('Timeout while waiting for import task Id %s' % import_task_id)
        res = self.ex_describe_import_snapshot_tasks(import_task_id)
        snapshotId = res.snapshotId
        if snapshotId is None:
            time.sleep(interval)
    volumeSnapshot = VolumeSnapshot(snapshotId, driver=self)
    return volumeSnapshot