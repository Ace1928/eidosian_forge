import os
from .start_hook_base import RayOnSparkStartHook
from .utils import get_spark_session
import logging
import threading
import time

    This helper function create a proxy URL for databricks driver webapp forwarding.
    In databricks runtime, user does not have permission to directly access web
    service binding on driver machine port, but user can visit it by a proxy URL with
    following format: "/driver-proxy/o/{orgId}/{clusterId}/{port}/".
    