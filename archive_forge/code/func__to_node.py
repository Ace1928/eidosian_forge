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
def _to_node(self, data):
    """
        Convert an API node data object to a `Node` object
        """
    ID = data['metadata']['uid']
    name = data['metadata']['name']
    driver = self.connection.driver
    memory = data['status'].get('capacity', {}).get('memory', '0K')
    cpu = data['status'].get('capacity', {}).get('cpu', '1')
    if isinstance(cpu, str) and (not cpu.isnumeric()):
        cpu = to_n_cpus(cpu)
    image_name = data['status']['nodeInfo'].get('osImage')
    image = NodeImage(image_name, image_name, driver)
    size_name = f'{cpu} vCPUs, {memory} Ram'
    size_id = hashlib.md5(size_name.encode('utf-8')).hexdigest()
    extra_size = {'cpus': cpu}
    size = NodeSize(id=size_id, name=size_name, ram=memory, disk=0, bandwidth=0, price=0, driver=driver, extra=extra_size)
    extra = {'memory': memory, 'cpu': cpu}
    extra['os'] = data['status']['nodeInfo'].get('operatingSystem')
    extra['kubeletVersion'] = data['status']['nodeInfo']['kubeletVersion']
    extra['provider_id'] = data.get('spec', {}).get('providerID')
    for condition in data['status']['conditions']:
        if condition['type'] == 'Ready' and condition['status'] == 'True':
            state = NodeState.RUNNING
            break
    else:
        state = NodeState.UNKNOWN
    public_ips, private_ips = ([], [])
    for address in data['status']['addresses']:
        if address['type'] == 'InternalIP':
            private_ips.append(address['address'])
        elif address['type'] == 'ExternalIP':
            public_ips.append(address['address'])
    created_at = datetime.datetime.strptime(data['metadata']['creationTimestamp'], '%Y-%m-%dT%H:%M:%SZ')
    return Node(id=ID, name=name, state=state, public_ips=public_ips, private_ips=private_ips, driver=driver, image=image, size=size, extra=extra, created_at=created_at)