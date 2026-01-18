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
def _deployment_info_to_schema(name: str, info: DeploymentInfo) -> DeploymentSchema:
    """Converts a DeploymentInfo object to DeploymentSchema.

    Route_prefix will not be set in the returned DeploymentSchema, since starting in 2.x
    route_prefix is an application-level concept. (This should only be used on the 2.x
    codepath)
    """
    schema = DeploymentSchema(name=name, max_concurrent_queries=info.deployment_config.max_concurrent_queries, user_config=info.deployment_config.user_config, graceful_shutdown_wait_loop_s=info.deployment_config.graceful_shutdown_wait_loop_s, graceful_shutdown_timeout_s=info.deployment_config.graceful_shutdown_timeout_s, health_check_period_s=info.deployment_config.health_check_period_s, health_check_timeout_s=info.deployment_config.health_check_timeout_s, ray_actor_options=info.replica_config.ray_actor_options)
    if info.deployment_config.autoscaling_config is not None:
        schema.autoscaling_config = info.deployment_config.autoscaling_config
    else:
        schema.num_replicas = info.deployment_config.num_replicas
    return schema