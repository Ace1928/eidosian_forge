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
class GCEAddress(UuidMixin):
    """A GCE Static address."""

    def __init__(self, id, name, address, region, driver, extra=None):
        self.id = str(id)
        self.name = name
        self.address = address
        self.region = region
        self.driver = driver
        self.extra = extra
        UuidMixin.__init__(self)

    def destroy(self):
        """
        Destroy this address.

        :return: True if successful
        :rtype:  ``bool``
        """
        return self.driver.ex_destroy_address(address=self)

    def __repr__(self):
        return '<GCEAddress id="{}" name="{}" address="{}" region="{}">'.format(self.id, self.name, self.address, hasattr(self.region, 'name') and self.region.name or self.region)