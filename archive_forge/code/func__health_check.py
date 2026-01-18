import json
import logging
import os
import random
import traceback
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Type
import ray
from ray.actor import ActorHandle
from ray.exceptions import RayActorError
from ray.serve._private.cluster_node_info_cache import ClusterNodeInfoCache
from ray.serve._private.common import NodeId, ProxyStatus
from ray.serve._private.constants import (
from ray.serve._private.proxy import ProxyActor
from ray.serve._private.utils import Timer, TimerBase, format_actor_name
from ray.serve.config import DeploymentMode, HTTPOptions, gRPCOptions
from ray.serve.schema import LoggingConfig, ProxyDetails
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def _health_check(self):
    """Perform periodic health checks."""
    assert self._status in {ProxyStatus.HEALTHY, ProxyStatus.DRAINING}
    if self._actor_proxy_wrapper.health_check_ongoing:
        try:
            healthy_call_status = self._actor_proxy_wrapper.is_healthy()
            if healthy_call_status == ProxyWrapperCallStatus.FINISHED_SUCCEED:
                self.try_update_status(self._status)
            elif healthy_call_status == ProxyWrapperCallStatus.FINISHED_FAILED:
                self.try_update_status(ProxyStatus.UNHEALTHY)
            elif self._timer.time() - self._last_health_check_time > PROXY_HEALTH_CHECK_TIMEOUT_S:
                self._actor_proxy_wrapper.reset_health_check()
                logger.warning(f"Didn't receive health check response for proxy {self._node_id} after {PROXY_HEALTH_CHECK_TIMEOUT_S}s")
                self.try_update_status(ProxyStatus.UNHEALTHY)
        except Exception as e:
            logger.warning(f'Health check for proxy {self._actor_name} failed: {e}')
            self.try_update_status(ProxyStatus.UNHEALTHY)
    if self._actor_proxy_wrapper.health_check_ongoing:
        return
    randomized_period_s = PROXY_HEALTH_CHECK_PERIOD_S * random.uniform(0.9, 1.1)
    if self._timer.time() - self._last_health_check_time > randomized_period_s:
        self._last_health_check_time = self._timer.time()
        self._actor_proxy_wrapper.start_new_health_check()