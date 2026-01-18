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
def _to_namespace(self, data):
    """
        Convert an API node data object to a `KubernetesNamespace` object
        """
    return KubernetesNamespace(id=data['metadata']['name'], name=data['metadata']['name'], driver=self.connection.driver, extra={'phase': data['status']['phase']})