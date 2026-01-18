import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from typing import Union
import ray._private.ray_constants as ray_constants
import ray._private.utils as utils
import ray.dashboard.consts as dashboard_consts
import ray.dashboard.utils as dashboard_utils
from ray.core.generated import event_pb2, event_pb2_grpc
from ray.dashboard.modules.event import event_consts
from ray.dashboard.modules.event.event_utils import monitor_events
from ray.dashboard.utils import async_loop_forever, create_task
Report events from cached events queue. Reconnect to dashboard if
        report failed. Log error after retry EVENT_AGENT_RETRY_TIMES.

        This method will never returns.
        