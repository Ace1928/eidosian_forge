import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
class GCEBackend(UuidMixin):
    """A GCE Backend.  Only used for creating Backend Services."""

    def __init__(self, instance_group, balancing_mode='UTILIZATION', max_utilization=None, max_rate=None, max_rate_per_instance=None, capacity_scaler=1, description=None):
        if isinstance(instance_group, GCEInstanceGroup):
            self.instance_group = instance_group
        elif isinstance(instance_group, GCEInstanceGroupManager):
            self.instance_group = instance_group.instance_group
        else:
            raise ValueError('instance_group must be of type GCEInstanceGroupor of type GCEInstanceGroupManager')
        self.instance_group = instance_group
        self.balancing_mode = balancing_mode
        self.max_utilization = max_utilization
        self.max_rate = max_rate
        self.max_rate_per_instance = max_rate_per_instance
        self.capacity_scaler = capacity_scaler
        self.id = self._gen_id()
        self.name = self.id
        self.description = description or self.name
        UuidMixin.__init__(self)

    def _gen_id(self):
        """
        Use the Instance Group information to fill in name and id fields.

        :return: id in the format of:
                 ZONE/instanceGroups/INSTANCEGROUPNAME
                 Ex: us-east1-c/instanceGroups/my-instance-group
        :rtype:  ``str``
        """
        zone_name = self.instance_group.zone.name
        return '{}/instanceGroups/{}'.format(zone_name, self.instance_group.name)

    def to_backend_dict(self):
        """
        Returns dict formatted for inclusion in Backend Service Request.

        :return: dict formatted as a list entry for Backend Service 'backend'.
        :rtype: ``dict``
        """
        d = {}
        d['group'] = self.instance_group.extra['selfLink']
        if self.balancing_mode:
            d['balancingMode'] = self.balancing_mode
        if self.max_utilization:
            d['maxUtilization'] = self.max_utilization
        if self.max_rate:
            d['maxRate'] = self.max_rate
        if self.max_rate_per_instance:
            d['maxRatePerInstance'] = self.max_rate_per_instance
        if self.capacity_scaler:
            d['capacityScaler'] = self.capacity_scaler
        return d

    def __repr__(self):
        return '<GCEBackend instancegroup="{}" balancing_mode="{}">'.format(self.id, self.balancing_mode)