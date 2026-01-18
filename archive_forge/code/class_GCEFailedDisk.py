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
class GCEFailedDisk:
    """Dummy Node object for disks that are not created."""

    def __init__(self, name, error, code):
        self.name = name
        self.error = error
        self.code = code

    def __repr__(self):
        return '<GCEFailedDisk name="{}" error_code="{}">'.format(self.name, self.code)