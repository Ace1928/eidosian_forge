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
class GCETargetInstance(UuidMixin):

    def __init__(self, id, name, zone, node, driver, extra=None):
        self.id = str(id)
        self.name = name
        self.zone = zone
        self.node = node
        self.driver = driver
        self.extra = extra
        UuidMixin.__init__(self)

    def destroy(self):
        """
        Destroy this Target Instance

        :return:  True if successful
        :rtype:   ``bool``
        """
        return self.driver.ex_destroy_targetinstance(targetinstance=self)

    def __repr__(self):
        return '<GCETargetInstance id="{}" name="{}" zone="{}" node="{}">'.format(self.id, self.name, self.zone.name, hasattr(self.node, 'name') and self.node.name or self.node)