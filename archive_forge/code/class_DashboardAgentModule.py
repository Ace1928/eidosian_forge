import abc
import asyncio
import datetime
import functools
import importlib
import json
import logging
import os
import pkgutil
from abc import ABCMeta, abstractmethod
from base64 import b64decode
from collections import namedtuple
from collections.abc import MutableMapping, Mapping, Sequence
from typing import Optional
import ray
import ray._private.ray_constants as ray_constants
import ray._private.services as services
from ray._raylet import GcsClient
from ray._private.utils import split_address
import aiosignal  # noqa: F401
import ray._private.protobuf_compat
from frozenlist import FrozenList  # noqa: F401
from ray._private.utils import binary_to_hex, check_dashboard_dependencies_installed
class DashboardAgentModule(abc.ABC):

    def __init__(self, dashboard_agent):
        """
        Initialize current module when DashboardAgent loading modules.
        :param dashboard_agent: The DashboardAgent instance.
        """
        self._dashboard_agent = dashboard_agent

    @abc.abstractmethod
    async def run(self, server):
        """
        Run the module in an asyncio loop. An agent module can provide
        servicers to the server.
        :param server: Asyncio GRPC server, or None if ray is minimal.
        """

    @staticmethod
    @abc.abstractclassmethod
    def is_minimal_module():
        """
        Return True if the module is minimal, meaning it
        should work with `pip install ray` that doesn't requires additional
        dependencies.
        """

    def get_gcs_address(self):
        return self._dashboard_agent.gcs_address