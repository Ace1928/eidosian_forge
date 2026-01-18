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
class ActorProxyWrapper(ProxyWrapper):

    def __init__(self, logging_config: LoggingConfig, actor_handle: Optional[ActorHandle]=None, config: Optional[HTTPOptions]=None, grpc_options: Optional[gRPCOptions]=None, controller_name: Optional[str]=None, name: Optional[str]=None, node_id: Optional[str]=None, node_ip_address: Optional[str]=None, port: Optional[int]=None, proxy_actor_class: Type[ProxyActor]=ProxyActor):
        self._actor_handle = actor_handle or self._get_or_create_proxy_actor(config=config, grpc_options=grpc_options, controller_name=controller_name, name=name, node_id=node_id, node_ip_address=node_ip_address, port=port, proxy_actor_class=proxy_actor_class, logging_config=logging_config)
        self._ready_obj_ref = None
        self._health_check_obj_ref = None
        self._is_drained_obj_ref = None
        self._update_draining_obj_ref = None
        self.worker_id = None
        self.log_file_path = None

    @staticmethod
    def _get_or_create_proxy_actor(config: HTTPOptions, grpc_options: gRPCOptions, controller_name: str, name: str, node_id: str, node_ip_address: str, port: int, logging_config: LoggingConfig, proxy_actor_class: Type[ProxyActor]=ProxyActor) -> ProxyWrapper:
        """Helper to start or reuse existing proxy.

        Takes the name of the proxy, the node id, and the node ip address, and look up
        or creates a new ProxyActor actor handle for the proxy.
        """
        proxy = None
        try:
            proxy = ray.get_actor(name, namespace=SERVE_NAMESPACE)
        except ValueError:
            logger.info(f"Starting proxy with name '{name}' on node '{node_id}' listening on '{config.host}:{port}'", extra={'log_to_stderr': False})
        proxy = proxy or proxy_actor_class.options(num_cpus=config.num_cpus, name=name, namespace=SERVE_NAMESPACE, lifetime='detached', max_concurrency=ASYNC_CONCURRENCY, max_restarts=0, scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False)).remote(config.host, port, config.root_path, controller_name=controller_name, node_ip_address=node_ip_address, node_id=node_id, http_middlewares=config.middlewares, request_timeout_s=config.request_timeout_s, keep_alive_timeout_s=config.keep_alive_timeout_s, grpc_options=grpc_options, logging_config=logging_config)
        return proxy

    @property
    def actor_id(self) -> str:
        """Return the actor id of the proxy actor."""
        return self._actor_handle._actor_id.hex()

    @property
    def actor_handle(self) -> ActorHandle:
        """Return the actor handle of the proxy actor.

        This is used in _start_controller() in _private/controller.py to check whether
        the proxies exist. It is also used in some tests to access proxy's actor handle.
        """
        return self._actor_handle

    @property
    def health_check_ongoing(self) -> bool:
        """Return whether the health check is ongoing or not."""
        return self._health_check_obj_ref is not None

    @property
    def is_draining(self) -> bool:
        """Return whether the drained check is ongoing or not."""
        return self._is_drained_obj_ref is not None

    def reset_drained_check(self):
        """Reset the drained check object reference."""
        self._is_drained_obj_ref = None

    def reset_health_check(self):
        """Reset the health check object reference."""
        self._health_check_obj_ref = None

    def start_new_ready_check(self):
        """Start a new ready check on the proxy actor."""
        self._ready_obj_ref = self._actor_handle.ready.remote()

    def start_new_health_check(self):
        """Start a new health check on the proxy actor."""
        self._health_check_obj_ref = self._actor_handle.check_health.remote()

    def start_new_drained_check(self):
        """Start a new drained check on the proxy actor.

        This is triggered once the proxy actor is set to draining. We will leave some
        time padding for the proxy actor to finish the ongoing requests. Once all
        ongoing requests are finished and the minimum draining time is met, the proxy
        actor will be transition to drained state and ready to be killed.
        """
        self._is_drained_obj_ref = self._actor_handle.is_drained.remote(_after=self._update_draining_obj_ref)

    def is_ready(self) -> ProxyWrapperCallStatus:
        """Return the payload from proxy ready check when ready.

        If the ongoing ready check is finished, and the value can be retrieved and
        unpacked, set the worker_id and log_file_path attributes of the proxy actor
        and return FINISHED_SUCCEED status. If the ongoing ready check is not finished,
        return PENDING status. If the RayActorError is raised, meaning that the actor
        is dead, return FINISHED_FAILED status.
        """
        try:
            finished, _ = ray.wait([self._ready_obj_ref], timeout=0)
            if finished:
                worker_id, log_file_path = json.loads(ray.get(finished[0]))
                self.worker_id = worker_id
                self.log_file_path = log_file_path
                return ProxyWrapperCallStatus.FINISHED_SUCCEED
            else:
                return ProxyWrapperCallStatus.PENDING
        except RayActorError:
            return ProxyWrapperCallStatus.FINISHED_FAILED

    def is_healthy(self) -> ProxyWrapperCallStatus:
        """Return whether the proxy actor is healthy or not.

        If the ongoing health check is finished, and the value can be retrieved,
        reset _health_check_obj_ref to enable the next health check and return
        FINISHED_SUCCEED status. If the ongoing ready check is not finished,
        return PENDING status. If the RayActorError is raised, meaning that the actor
        is dead, return FINISHED_FAILED status.
        """
        try:
            finished, _ = ray.wait([self._health_check_obj_ref], timeout=0)
            if finished:
                self._health_check_obj_ref = None
                ray.get(finished[0])
                return ProxyWrapperCallStatus.FINISHED_SUCCEED
            else:
                return ProxyWrapperCallStatus.PENDING
        except RayActorError:
            return ProxyWrapperCallStatus.FINISHED_FAILED

    def is_drained(self) -> ProxyWrapperCallStatus:
        """Return whether the proxy actor is drained or not.

        If the ongoing drained check is finished, and the value can be retrieved,
        reset _is_drained_obj_ref to ensure drained check is finished and return
        FINISHED_SUCCEED status. If the ongoing ready check is not finished,
        return PENDING status.
        """
        finished, _ = ray.wait([self._is_drained_obj_ref], timeout=0)
        if finished:
            self._is_drained_obj_ref = None
            is_drained = ray.get(finished[0])
            if is_drained:
                return ProxyWrapperCallStatus.FINISHED_SUCCEED
            else:
                return ProxyWrapperCallStatus.FINISHED_FAILED
        else:
            return ProxyWrapperCallStatus.PENDING

    def is_shutdown(self) -> bool:
        """Return whether the proxy actor is shutdown.

        If the actor is dead, the health check will return RayActorError.
        """
        try:
            ray.get(self._actor_handle.check_health.remote(), timeout=0)
        except RayActorError:
            return True
        return False

    def update_draining(self, draining: bool):
        """Update the draining status of the proxy actor."""
        self._update_draining_obj_ref = self._actor_handle.update_draining.remote(draining, _after=self._update_draining_obj_ref)

    def kill(self):
        """Kill the proxy actor."""
        ray.kill(self._actor_handle, no_restart=True)