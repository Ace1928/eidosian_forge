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
class GCEAutoscaler(UuidMixin):
    """Represents a autoscaling policy object used to scale Instance Groups."""

    def __init__(self, id, name, zone, target, policy, driver, extra=None):
        self.id = str(id)
        self.name = name
        self.zone = zone
        self.target = target
        self.policy = policy
        self.driver = driver
        self.extra = extra
        UuidMixin.__init__(self)

    def destroy(self):
        """
        Destroy this Autoscaler.

        :return:  True if successful
        :rtype:   ``bool``
        """
        return self.driver.ex_destroy_autoscaler(autoscaler=self)

    def __repr__(self):
        return '<GCEAutoScaler id="{}" name="{}" zone="{}" target="{}">'.format(self.id, self.name, self.zone.name, self.target.name)