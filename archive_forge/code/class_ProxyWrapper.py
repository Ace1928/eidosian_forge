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
class ProxyWrapper(ABC):

    @property
    @abstractmethod
    def actor_id(self) -> str:
        """Return the actor id of the proxy actor."""
        raise NotImplementedError

    @abstractmethod
    def start_new_ready_check(self):
        """Start a new ready check on the proxy actor."""
        raise NotImplementedError

    @abstractmethod
    def start_new_health_check(self):
        """Start a new health check on the proxy actor."""
        raise NotImplementedError

    @abstractmethod
    def start_new_drained_check(self):
        """Start a new drained check on the proxy actor.

        This is triggered once the proxy actor is set to draining. We will leave some
        time padding for the proxy actor to finish the ongoing requests. Once all
        ongoing requests are finished and the minimum draining time is met, the proxy
        actor will be transition to drained state and ready to be killed.
        """
        raise NotImplementedError

    @abstractmethod
    def is_ready(self) -> ProxyWrapperCallStatus:
        """Return the payload from proxy ready check when ready."""
        raise NotImplementedError

    @abstractmethod
    def is_healthy(self) -> ProxyWrapperCallStatus:
        """Return whether the proxy actor is healthy or not."""
        raise NotImplementedError

    @abstractmethod
    def is_drained(self) -> ProxyWrapperCallStatus:
        """Return whether the proxy actor is drained or not."""
        raise NotImplementedError

    @abstractmethod
    def is_shutdown(self):
        """Return whether the proxy actor is shutdown."""
        raise NotImplementedError

    @abstractmethod
    def update_draining(self, draining: bool):
        """Update the draining status of the proxy actor."""
        raise NotImplementedError

    @abstractmethod
    def kill(self):
        """Kill the proxy actor."""
        raise NotImplementedError