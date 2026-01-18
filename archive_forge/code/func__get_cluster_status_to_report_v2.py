import json
import logging
import threading
import os
import platform
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set
import requests
import yaml
import ray
import ray._private.ray_constants as ray_constants
import ray._private.usage.usage_constants as usage_constant
from ray.experimental.internal_kv import _internal_kv_initialized, _internal_kv_put
from ray.core.generated import usage_pb2, gcs_pb2
def _get_cluster_status_to_report_v2(gcs_client) -> ClusterStatusToReport:
    """
    Get the current status of this cluster. A temporary proxy for the
    autoscaler v2 API.

    It is a blocking API.

    Params:
        gcs_client: The GCS client.

    Returns:
        The current cluster status or empty ClusterStatusToReport
        if it fails to get that information.
    """
    from ray.autoscaler.v2.sdk import get_cluster_status
    result = ClusterStatusToReport()
    try:
        cluster_status = get_cluster_status(gcs_client.address)
        total_resources = cluster_status.total_resources()
        result.total_num_cpus = total_resources.get('CPU', 0)
        result.total_num_gpus = total_resources.get('GPU', 0)
        to_GiB = 1 / 2 ** 30
        result.total_memory_gb = total_resources.get('memory', 0) * to_GiB
        result.total_object_store_memory_gb = total_resources.get('object_store_memory', 0) * to_GiB
    except Exception as e:
        logger.info(f'Failed to get cluster status to report {e}')
    finally:
        return result