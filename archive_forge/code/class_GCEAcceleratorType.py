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
class GCEAcceleratorType(UuidMixin):
    """A GCE AcceleratorType resource."""

    def __init__(self, id, name, zone, driver, extra=None):
        self.id = str(id)
        self.name = name
        self.zone = zone
        self.driver = driver
        self.extra = extra
        UuidMixin.__init__(self)

    def destroy(self):
        raise LibcloudError('Can not destroy an AcceleratorType resource.')

    def __repr__(self):
        return '<GCEAcceleratorType id="{}" name="{}" zone="{}">'.format(self.id, self.name, self.zone)