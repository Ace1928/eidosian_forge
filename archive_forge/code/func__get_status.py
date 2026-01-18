import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from zlib import crc32
from ray._private.pydantic_compat import (
from ray._private.runtime_env.packaging import parse_uri
from ray.serve._private.common import (
from ray.serve._private.constants import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.utils import DEFAULT
from ray.serve.config import ProxyLocation
from ray.util.annotations import PublicAPI
def _get_status(self) -> ServeStatus:
    return ServeStatus(target_capacity=self.target_capacity, proxies={node_id: proxy.status for node_id, proxy in self.proxies.items()}, applications={app_name: ApplicationStatusOverview(status=app.status, message=app.message, last_deployed_time_s=app.last_deployed_time_s, deployments={deployment_name: DeploymentStatusOverview(status=deployment.status, status_trigger=deployment.status_trigger, replica_states=dict(Counter([r.state.value for r in deployment.replicas])), message=deployment.message) for deployment_name, deployment in app.deployments.items()}) for app_name, app in self.applications.items()})