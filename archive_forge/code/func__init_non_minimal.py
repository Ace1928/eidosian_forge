import argparse
import asyncio
import json
import logging
import logging.handlers
import os
import pathlib
import sys
import signal
import ray
import ray._private.ray_constants as ray_constants
import ray._private.services
import ray._private.utils
import ray.dashboard.consts as dashboard_consts
import ray.dashboard.utils as dashboard_utils
from ray._raylet import GcsClient
from ray._private.process_watcher import create_check_raylet_task
from ray._private.gcs_utils import GcsAioClient
from ray._private.ray_logging import (
from ray.experimental.internal_kv import (
from ray._private.ray_constants import AGENT_GRPC_MAX_MESSAGE_LENGTH
def _init_non_minimal(self):
    from ray._private.gcs_pubsub import GcsAioPublisher
    self.aio_publisher = GcsAioPublisher(address=self.gcs_address)
    try:
        from grpc import aio as aiogrpc
    except ImportError:
        from grpc.experimental import aio as aiogrpc
    if sys.version_info.major >= 3 and sys.version_info.minor >= 10:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            aiogrpc.init_grpc_aio()
    else:
        aiogrpc.init_grpc_aio()
    self.server = aiogrpc.server(options=(('grpc.so_reuseport', 0), ('grpc.max_send_message_length', AGENT_GRPC_MAX_MESSAGE_LENGTH), ('grpc.max_receive_message_length', AGENT_GRPC_MAX_MESSAGE_LENGTH)))
    grpc_ip = '127.0.0.1' if self.ip == '127.0.0.1' else '0.0.0.0'
    try:
        self.grpc_port = ray._private.tls_utils.add_port_to_grpc_server(self.server, f'{grpc_ip}:{self.dashboard_agent_port}')
    except Exception:
        logger.exception('Failed to add port to grpc server. Agent will stay alive but disable the grpc service.')
        self.server = None
        self.grpc_port = None
    else:
        logger.info('Dashboard agent grpc address: %s:%s', grpc_ip, self.grpc_port)