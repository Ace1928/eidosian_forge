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
def _to_forwarding_rule(self, forwarding_rule):
    """
        Return a Forwarding Rule object from the JSON-response dictionary.

        :param  forwarding_rule: The dictionary describing the rule.
        :type   forwarding_rule: ``dict``

        :return: ForwardingRule object
        :rtype: :class:`GCEForwardingRule`
        """
    extra = {}
    extra['selfLink'] = forwarding_rule.get('selfLink')
    extra['portRange'] = forwarding_rule.get('portRange')
    extra['creationTimestamp'] = forwarding_rule.get('creationTimestamp')
    extra['description'] = forwarding_rule.get('description')
    region = forwarding_rule.get('region')
    if region:
        region = self.ex_get_region(region)
    target = self._get_object_by_kind(forwarding_rule['target'])
    return GCEForwardingRule(id=forwarding_rule['id'], name=forwarding_rule['name'], region=region, address=forwarding_rule.get('IPAddress'), protocol=forwarding_rule.get('IPProtocol'), targetpool=target, driver=self, extra=extra)