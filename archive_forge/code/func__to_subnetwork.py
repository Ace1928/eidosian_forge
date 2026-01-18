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
def _to_subnetwork(self, subnetwork):
    """
        Return a Subnetwork object from the JSON-response dictionary.

        :param  subnetwork: The dictionary describing the subnetwork.
        :type   subnetwork: ``dict``

        :return: Subnetwork object
        :rtype: :class:`GCESubnetwork`
        """
    extra = {}
    extra['creationTimestamp'] = subnetwork.get('creationTimestamp')
    extra['description'] = subnetwork.get('description')
    extra['gatewayAddress'] = subnetwork.get('gatewayAddress')
    extra['ipCidrRange'] = subnetwork.get('ipCidrRange')
    extra['network'] = subnetwork.get('network')
    extra['region'] = subnetwork.get('region')
    extra['selfLink'] = subnetwork.get('selfLink')
    extra['privateIpGoogleAccess'] = subnetwork.get('privateIpGoogleAccess')
    extra['secondaryIpRanges'] = subnetwork.get('secondaryIpRanges')
    network = self._get_object_by_kind(subnetwork.get('network'))
    region = self._get_object_by_kind(subnetwork.get('region'))
    return GCESubnetwork(id=subnetwork['id'], name=subnetwork['name'], cidr=subnetwork.get('ipCidrRange'), network=network, region=region, driver=self, extra=extra)