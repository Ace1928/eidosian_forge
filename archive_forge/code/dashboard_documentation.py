import argparse
import logging
import logging.handlers
import platform
import traceback
import signal
import os
import sys
import ray._private.ray_constants as ray_constants
import ray._private.services
import ray._private.utils
import ray.dashboard.consts as dashboard_consts
import ray.dashboard.head as dashboard_head
import ray.dashboard.utils as dashboard_utils
from ray._private.ray_logging import setup_component_logger
from typing import Optional, Set
A dashboard process for monitoring Ray nodes.

    This dashboard is made up of a REST API which collates data published by
        Reporter processes on nodes into a json structure, and a webserver
        which polls said API for display purposes.

    Args:
        host: Host address of dashboard aiohttp server.
        port: Port number of dashboard aiohttp server.
        port_retries: The retry times to select a valid port.
        gcs_address: GCS address of the cluster
        grpc_port: Port used to listen for gRPC on.
        node_ip_address: The IP address of the dashboard.
        serve_frontend: If configured, frontend HTML
            is not served from the dashboard.
        log_dir: Log directory of dashboard.
    