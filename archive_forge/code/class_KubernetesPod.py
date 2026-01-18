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
class KubernetesPod:

    def __init__(self, id: str, name: str, containers: List[Container], namespace: str, state: str, ip_addresses: List[str], created_at: datetime.datetime, node_name: str, extra: Dict[str, Any]):
        """
        A Kubernetes pod
        """
        self.id = id
        self.name = name
        self.containers = containers
        self.namespace = namespace
        self.state = state
        self.ip_addresses = ip_addresses
        self.created_at = created_at
        self.node_name = node_name
        self.extra = extra

    def __repr__(self):
        return '<KubernetesPod name={} namespace={} state={}>'.format(self.name, self.namespace, self.state)