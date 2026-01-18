import asyncio
from datetime import datetime
import inspect
import fnmatch
import functools
import io
import json
import logging
import math
import os
import pathlib
import random
import socket
import subprocess
import sys
import tempfile
import time
import timeit
import traceback
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict, List, Optional
import uuid
from dataclasses import dataclass
import requests
from ray._raylet import Config
import psutil  # We must import psutil after ray because we bundle it with ray.
from ray._private import (
from ray._private.worker import RayContext
import yaml
import ray
import ray._private.gcs_utils as gcs_utils
import ray._private.memory_monitor as memory_monitor
import ray._private.services
import ray._private.utils
from ray._private.internal_api import memory_summary
from ray._private.tls_utils import generate_self_signed_tls_certs
from ray._raylet import GcsClientOptions, GlobalStateAccessor
from ray.core.generated import (
from ray.util.queue import Empty, Queue, _QueueActor
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
@ray.remote(num_cpus=0)
class NodeKillerActor(ResourceKillerActor):

    async def _find_resource_to_kill(self):
        node_to_kill_ip = None
        node_to_kill_port = None
        node_id = None
        while node_to_kill_port is None and self.is_running:
            nodes = ray.nodes()
            alive_nodes = self._get_alive_nodes(nodes)
            if self.kill_filter_fn is not None:
                nodes = list(filter(self.kill_filter_fn(), nodes))
            for node in nodes:
                node_id = node['NodeID']
                if node['Alive'] and node_id != self.head_node_id and (node_id not in self.killed) and (alive_nodes > 2):
                    node_to_kill_ip = node['NodeManagerAddress']
                    node_to_kill_port = node['NodeManagerPort']
                    break
            await asyncio.sleep(0.1)
        return (node_id, node_to_kill_ip, node_to_kill_port)

    def _kill_resource(self, node_id, node_to_kill_ip, node_to_kill_port):
        if node_to_kill_port is not None:
            try:
                self._kill_raylet(node_to_kill_ip, node_to_kill_port, graceful=False)
            except Exception:
                pass
            logging.info(f'Killed node {node_id} at address: {node_to_kill_ip}, port: {node_to_kill_port}')
            self.killed.add(node_id)

    def _kill_raylet(self, ip, port, graceful=False):
        import grpc
        from grpc._channel import _InactiveRpcError
        from ray.core.generated import node_manager_pb2_grpc
        raylet_address = f'{ip}:{port}'
        channel = grpc.insecure_channel(raylet_address)
        stub = node_manager_pb2_grpc.NodeManagerServiceStub(channel)
        try:
            stub.ShutdownRaylet(node_manager_pb2.ShutdownRayletRequest(graceful=graceful))
        except _InactiveRpcError:
            assert not graceful

    def _get_alive_nodes(self, nodes):
        alive_nodes = 0
        for node in nodes:
            if node['Alive']:
                alive_nodes += 1
        return alive_nodes