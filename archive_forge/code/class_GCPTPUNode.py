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
class GCPTPUNode(GCPNode):
    """Abstraction around tpu nodes"""
    NON_TERMINATED_STATUSES = {'CREATING', 'STARTING', 'RESTARTING', 'READY'}
    RUNNING_STATUSES = {'READY'}
    STATUS_FIELD = 'state'

    def get_labels(self) -> dict:
        return self.get('labels', {})

    @property
    def num_workers(self) -> int:
        return len(self.get('networkEndpoints', [{}]))

    def get_external_ips(self) -> List[str]:
        return self.get('networkEndpoints', [{}])

    def get_external_ip(self, worker_index: int=0) -> str:
        return self.get_external_ips()[worker_index].get('accessConfig', {}).get('externalIp', None)

    def get_internal_ips(self) -> List[str]:
        return self.get('networkEndpoints', [{}])

    def get_internal_ip(self, worker_index: int=0) -> str:
        return self.get_internal_ips()[worker_index].get('ipAddress', None)