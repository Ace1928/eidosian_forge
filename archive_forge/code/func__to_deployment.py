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
def _to_deployment(self, data):
    id_ = data['metadata']['uid']
    name = data['metadata']['name']
    namespace = data['metadata']['namespace']
    created_at = data['metadata']['creationTimestamp']
    replicas = data['spec']['replicas']
    selector = data['spec']['selector']
    extra = {'labels': data['metadata']['labels'], 'strategy': data['spec']['strategy']['type'], 'total_replicas': data['status']['replicas'], 'updated_replicas': data['status']['updatedReplicas'], 'ready_replicas': data['status']['readyReplicas'], 'available_replicas': data['status']['availableReplicas'], 'conditions': data['status']['conditions']}
    return KubernetesDeployment(id=id_, name=name, namespace=namespace, created_at=created_at, replicas=replicas, selector=selector, extra=extra)