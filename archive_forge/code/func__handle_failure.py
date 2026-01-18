import argparse
import json
import logging
import os
import signal
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional, Union
import ray
import ray._private.ray_constants as ray_constants
import ray._private.utils
from ray._private.event.event_logger import get_event_logger
from ray._private.ray_logging import setup_component_logger
from ray._raylet import GcsClient
from ray.autoscaler._private.autoscaler import StandardAutoscaler
from ray.autoscaler._private.commands import teardown_cluster
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.event_summarizer import EventSummarizer
from ray.autoscaler._private.load_metrics import LoadMetrics
from ray.autoscaler._private.prom_metrics import AutoscalerPrometheusMetrics
from ray.autoscaler._private.util import format_readonly_node_type
from ray.core.generated import gcs_pb2
from ray.core.generated.event_pb2 import Event as RayEvent
from ray.experimental.internal_kv import (
def _handle_failure(self, error):
    if self.autoscaler is not None and os.environ.get('RAY_AUTOSCALER_FATESHARE_WORKERS', '') == '1':
        self.autoscaler.kill_workers()
        self.destroy_autoscaler_workers()
    message = f'The autoscaler failed with the following error:\n{error}'
    if _internal_kv_initialized():
        _internal_kv_put(ray_constants.DEBUG_AUTOSCALING_ERROR, message, overwrite=True)
    gcs_publisher = ray._raylet.GcsPublisher(address=self.gcs_address)
    from ray._private.utils import publish_error_to_driver
    publish_error_to_driver(ray_constants.MONITOR_DIED_ERROR, message, gcs_publisher=gcs_publisher)