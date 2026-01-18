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
class GCENetwork(UuidMixin):
    """A GCE Network object class."""

    def __init__(self, id, name, cidr, driver, extra=None):
        self.id = str(id)
        self.name = name
        self.cidr = cidr
        self.driver = driver
        self.extra = extra
        self.mode = 'legacy'
        self.subnetworks = []
        if 'mode' in extra and extra['mode'] != 'legacy':
            self.mode = extra['mode']
            self.subnetworks = extra['subnetworks']
        UuidMixin.__init__(self)

    def destroy(self):
        """
        Destroy this network

        :return: True if successful
        :rtype:  ``bool``
        """
        return self.driver.ex_destroy_network(network=self)

    def __repr__(self):
        return '<GCENetwork id="{}" name="{}" cidr="{}" mode="{}">'.format(self.id, self.name, self.cidr, self.mode)