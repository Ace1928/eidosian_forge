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
def _to_firewall(self, firewall):
    """
        Return a Firewall object from the JSON-response dictionary.

        :param  firewall: The dictionary describing the firewall.
        :type   firewall: ``dict``

        :return: Firewall object
        :rtype: :class:`GCEFirewall`
        """
    extra = {}
    extra['selfLink'] = firewall.get('selfLink')
    extra['creationTimestamp'] = firewall.get('creationTimestamp')
    extra['description'] = firewall.get('description')
    extra['network_name'] = self._get_components_from_path(firewall['network'])['name']
    network = self.ex_get_network(extra['network_name'])
    allowed = firewall.get('allowed')
    denied = firewall.get('denied')
    priority = firewall.get('priority')
    direction = firewall.get('direction')
    source_ranges = firewall.get('sourceRanges')
    source_tags = firewall.get('sourceTags')
    source_service_accounts = firewall.get('sourceServiceAccounts')
    target_tags = firewall.get('targetTags')
    target_service_accounts = firewall.get('targetServiceAccounts')
    target_ranges = firewall.get('targetRanges')
    return GCEFirewall(id=firewall['id'], name=firewall['name'], allowed=allowed, denied=denied, network=network, target_ranges=target_ranges, source_ranges=source_ranges, priority=priority, source_tags=source_tags, target_tags=target_tags, source_service_accounts=source_service_accounts, target_service_accounts=target_service_accounts, direction=direction, driver=self, extra=extra)