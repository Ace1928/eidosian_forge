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
class GCESnapshot(VolumeSnapshot):

    def __init__(self, id, name, size, status, driver, extra=None, created=None):
        self.status = status
        super().__init__(id, driver, size, extra, created, name=name)