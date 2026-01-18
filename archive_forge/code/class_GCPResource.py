from API responses.
import abc
import logging
import re
import time
from collections import UserDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4
from googleapiclient.discovery import Resource
from googleapiclient.errors import HttpError
from ray.autoscaler.tags import TAG_RAY_CLUSTER_NAME, TAG_RAY_NODE_NAME
class GCPResource(metaclass=abc.ABCMeta):
    """Abstraction around compute and TPU resources"""

    def __init__(self, resource: Resource, project_id: str, availability_zone: str, cluster_name: str) -> None:
        self.resource = resource
        self.project_id = project_id
        self.availability_zone = availability_zone
        self.cluster_name = cluster_name

    @abc.abstractmethod
    def wait_for_operation(self, operation: dict, max_polls: int=MAX_POLLS, poll_interval: int=POLL_INTERVAL) -> dict:
        """Waits a preset amount of time for operation to complete."""
        return None

    @abc.abstractmethod
    def list_instances(self, label_filters: Optional[dict]=None, is_terminated: bool=False) -> List['GCPNode']:
        """Returns a filtered list of all instances.

        The filter removes all terminated instances and, if ``label_filters``
        are provided, all instances which labels are not matching the
        ones provided.
        """
        return

    @abc.abstractmethod
    def get_instance(self, node_id: str) -> 'GCPNode':
        """Returns a single instance."""
        return

    @abc.abstractmethod
    def set_labels(self, node: GCPNode, labels: dict, wait_for_operation: bool=True) -> dict:
        """Sets labels on an instance and returns result.

        Completely replaces the labels dictionary."""
        return

    @abc.abstractmethod
    def create_instance(self, base_config: dict, labels: dict, wait_for_operation: bool=True) -> Tuple[dict, str]:
        """Creates a single instance and returns result.

        Returns a tuple of (result, node_name).
        """
        return

    def create_instances(self, base_config: dict, labels: dict, count: int, wait_for_operation: bool=True) -> List[Tuple[dict, str]]:
        """Creates multiple instances and returns result.

        Returns a list of tuples of (result, node_name).
        """
        operations = [self.create_instance(base_config, labels, wait_for_operation=False) for i in range(count)]
        if wait_for_operation:
            results = [(self.wait_for_operation(operation), node_name) for operation, node_name in operations]
        else:
            results = operations
        return results

    @abc.abstractmethod
    def delete_instance(self, node_id: str, wait_for_operation: bool=True) -> dict:
        """Deletes an instance and returns result."""
        return

    @abc.abstractmethod
    def stop_instance(self, node_id: str, wait_for_operation: bool=True) -> dict:
        """Deletes an instance and returns result."""
        return

    @abc.abstractmethod
    def start_instance(self, node_id: str, wait_for_operation: bool=True) -> dict:
        """Starts a single instance and returns result."""
        return