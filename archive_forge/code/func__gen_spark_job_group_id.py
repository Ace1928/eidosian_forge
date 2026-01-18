import json
import logging
import sys
from threading import RLock
from typing import Any, Dict, Optional
import requests
from ray.autoscaler.node_launch_exception import NodeLaunchException
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def _gen_spark_job_group_id(self, node_id):
    return f'ray-cluster-{self.ray_head_port}-{self.cluster_id}-worker-node-{node_id}'