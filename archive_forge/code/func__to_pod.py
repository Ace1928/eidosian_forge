import json
import hashlib
import datetime
from typing import Any, Dict, List, Union, Optional
from collections import OrderedDict
from libcloud.compute.base import Node, NodeSize, NodeImage
from libcloud.compute.types import NodeState
from libcloud.container.base import Container, ContainerImage, ContainerDriver, ContainerCluster
from libcloud.container.types import ContainerState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.common.kubernetes import (
from libcloud.container.providers import Provider
def _to_pod(self, data, metrics=None):
    """
        Convert an API response to a Pod object
        """
    id_ = data['metadata']['uid']
    name = data['metadata']['name']
    namespace = data['metadata']['namespace']
    state = data['status']['phase'].lower()
    node_name = data['spec'].get('nodeName')
    container_statuses = data['status'].get('containerStatuses', {})
    containers = []
    extra = {'resources': {}}
    if metrics:
        try:
            extra['metrics'] = metrics[name, namespace]
        except KeyError:
            pass
    for container in data['spec']['containers']:
        if container_statuses:
            spec = list(filter(lambda i: i['name'] == container['name'], container_statuses))[0]
        else:
            spec = container_statuses
        container_obj = self._to_container(container, spec, data)
        resources = extra['resources']
        container_resources = container_obj.extra.get('resources', {})
        resources['limits'] = sum_resources(resources.get('limits', {}), container_resources.get('limits', {}))
        extra['resources']['requests'] = sum_resources(resources.get('requests', {}), container_resources.get('requests', {}))
        containers.append(container_obj)
    ip_addresses = [ip_dict['ip'] for ip_dict in data['status'].get('podIPs', [])]
    created_at = datetime.datetime.strptime(data['metadata']['creationTimestamp'], '%Y-%m-%dT%H:%M:%SZ')
    return KubernetesPod(id=id_, name=name, namespace=namespace, state=state, ip_addresses=ip_addresses, containers=containers, created_at=created_at, node_name=node_name, extra=extra)