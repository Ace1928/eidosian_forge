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
class GCEHealthCheck(UuidMixin):
    """A GCE Http Health Check class."""

    def __init__(self, id, name, path, port, interval, timeout, unhealthy_threshold, healthy_threshold, driver, extra=None):
        self.id = str(id)
        self.name = name
        self.path = path
        self.port = port
        self.interval = interval
        self.timeout = timeout
        self.unhealthy_threshold = unhealthy_threshold
        self.healthy_threshold = healthy_threshold
        self.driver = driver
        self.extra = extra or {}
        UuidMixin.__init__(self)

    def destroy(self):
        """
        Destroy this Health Check.

        :return:  True if successful
        :rtype:   ``bool``
        """
        return self.driver.ex_destroy_healthcheck(healthcheck=self)

    def update(self):
        """
        Commit updated healthcheck values.

        :return:  Updated Healthcheck object
        :rtype:   :class:`GCEHealthcheck`
        """
        return self.driver.ex_update_healthcheck(healthcheck=self)

    def __repr__(self):
        return '<GCEHealthCheck id="{}" name="{}" path="{}" port="{}">'.format(self.id, self.name, self.path, self.port)