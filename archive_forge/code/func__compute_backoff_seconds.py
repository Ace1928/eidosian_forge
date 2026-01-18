import enum
import functools
import random
import threading
import time
import grpc
from tensorboard import version
from tensorboard.util import tb_logging
def _compute_backoff_seconds(num_attempts):
    """Compute appropriate wait time between RPC attempts."""
    jitter_factor = random.uniform(_GRPC_RETRY_JITTER_FACTOR_MIN, _GRPC_RETRY_JITTER_FACTOR_MAX)
    backoff_secs = _GRPC_RETRY_EXPONENTIAL_BASE ** num_attempts * jitter_factor
    return backoff_secs