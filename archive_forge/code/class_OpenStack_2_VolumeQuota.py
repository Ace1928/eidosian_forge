import base64
import warnings
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, next, httplib, parse_qs, urlparse
from libcloud.utils.xml import findall
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.openstack import (
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
class OpenStack_2_VolumeQuota:
    """
    Volume Quota info. To get the information about quotas and used resources.

    See:
    https://docs.openstack.org/api-ref/block-storage/v2/index.html?expanded=show-quotas-detail
    https://docs.openstack.org/api-ref/block-storage/v3/index.html?expanded=show-quota-usage-for-a-project-detail
    """

    def __init__(self, backup_gigabytes, gigabytes, per_volume_gigabytes, backups, snapshots, volumes, driver=None):
        """
        :param backup_gigabytes: Quota of backup size in gigabytes.
        :type backup_gigabytes: :class:`.OpenStack_2_QuotaSetItem` or ``int``
        :param gigabytes: Quota of volume size in gigabytes.
        :type gigabytes: :class:`.OpenStack_2_QuotaSetItem` or ``int``
        :param per_volume_gigabytes: Quota of per volume gigabytes.
        :type per_volume_gigabytes: :class:`.OpenStack_2_QuotaSetItem`
                                    or ``int``
        :param backups: Quota of backups.
        :type backups: :class:`.OpenStack_2_QuotaSetItem` or ``int``
        :param snapshots: Quota of snapshots.
        :type snapshots: :class:`.OpenStack_2_QuotaSetItem` or ``int``
        :param volumes: Quota of security volumes.
        :type volumes: :class:`.OpenStack_2_QuotaSetItem` or ``int``
        """
        self.backup_gigabytes = self._to_quota_set_item(backup_gigabytes)
        self.gigabytes = self._to_quota_set_item(gigabytes)
        self.per_volume_gigabytes = self._to_quota_set_item(per_volume_gigabytes)
        self.backups = self._to_quota_set_item(backups)
        self.snapshots = self._to_quota_set_item(snapshots)
        self.volumes = self._to_quota_set_item(volumes)
        self.driver = driver

    def _to_quota_set_item(self, obj):
        if obj:
            if isinstance(obj, OpenStack_2_QuotaSetItem):
                return obj
            elif isinstance(obj, dict):
                return OpenStack_2_QuotaSetItem(obj['in_use'], obj['limit'], obj['reserved'])
            elif isinstance(obj, int):
                return OpenStack_2_QuotaSetItem(0, obj, 0)
            else:
                return None
        else:
            return None

    def __repr__(self):
        return '<OpenStack_2_VolumeQuota Volumes="%s", gigabytes="%s", snapshots="%s", backups="%s">' % (self.volumes, self.gigabytes, self.snapshots, self.backups)