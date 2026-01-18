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
class GCEFirewall(UuidMixin):
    """A GCE Firewall rule class."""

    def __init__(self, id, name, allowed, denied, direction, network, source_ranges, source_tags, priority, source_service_accounts, target_service_accounts, target_tags, target_ranges, driver, extra=None):
        self.id = str(id)
        self.name = name
        self.network = network
        self.allowed = allowed
        self.denied = denied
        self.direction = direction
        self.priority = priority
        self.source_ranges = source_ranges
        self.source_tags = source_tags
        self.source_service_accounts = source_service_accounts
        self.target_tags = target_tags
        self.target_service_accounts = target_service_accounts
        self.target_ranges = target_ranges
        self.driver = driver
        self.extra = extra
        UuidMixin.__init__(self)

    def destroy(self):
        """
        Destroy this firewall.

        :return: True if successful
        :rtype:  ``bool``
        """
        return self.driver.ex_destroy_firewall(firewall=self)

    def update(self):
        """
        Commit updated firewall values.

        :return:  Updated Firewall object
        :rtype:   :class:`GCEFirewall`
        """
        return self.driver.ex_update_firewall(firewall=self)

    def __repr__(self):
        return '<GCEFirewall id="{}" name="{}" network="{}">'.format(self.id, self.name, self.network.name)