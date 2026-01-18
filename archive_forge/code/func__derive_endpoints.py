import os
import platform
import socket
import sys
import futurist
from oslo_utils import reflection
from taskflow.engines.worker_based import endpoint
from taskflow.engines.worker_based import server
from taskflow import logging
from taskflow import task as t_task
from taskflow.utils import banner
from taskflow.utils import misc
from taskflow.utils import threading_utils as tu
@staticmethod
def _derive_endpoints(tasks):
    """Derive endpoints from list of strings, classes or packages."""
    derived_tasks = misc.find_subclasses(tasks, t_task.Task)
    return [endpoint.Endpoint(task) for task in derived_tasks]