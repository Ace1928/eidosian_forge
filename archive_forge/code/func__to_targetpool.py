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
def _to_targetpool(self, targetpool):
    """
        Return a Target Pool object from the JSON-response dictionary.

        :param  targetpool: The dictionary describing the volume.
        :type   targetpool: ``dict``

        :return: Target Pool object
        :rtype:  :class:`GCETargetPool`
        """
    extra = {}
    extra['selfLink'] = targetpool.get('selfLink')
    extra['description'] = targetpool.get('description')
    extra['sessionAffinity'] = targetpool.get('sessionAffinity')
    region = self.ex_get_region(targetpool['region'])
    healthcheck_list = [self.ex_get_healthcheck(h.split('/')[-1]) for h in targetpool.get('healthChecks', [])]
    node_list = []
    for n in targetpool.get('instances', []):
        comp = self._get_components_from_path(n)
        try:
            node = self.ex_get_node(comp['name'], comp['zone'])
        except ResourceNotFoundError:
            node = n
        node_list.append(node)
    if 'failoverRatio' in targetpool:
        extra['failoverRatio'] = targetpool['failoverRatio']
    if 'backupPool' in targetpool:
        tp_split = targetpool['backupPool'].split('/')
        extra['backupPool'] = self.ex_get_targetpool(tp_split[10], tp_split[8])
    return GCETargetPool(id=targetpool['id'], name=targetpool['name'], region=region, healthchecks=healthcheck_list, nodes=node_list, driver=self, extra=extra)