import datetime
import errno
import html
import json
import os
import random
import shlex
import textwrap
import time
from tensorboard import manager
def _time_delta_from_info(info):
    """Format the elapsed time for the given TensorBoardInfo.

    Args:
      info: A TensorBoardInfo value.

    Returns:
      A human-readable string describing the time since the server
      described by `info` started: e.g., "2 days, 0:48:58".
    """
    delta_seconds = int(time.time()) - info.start_time
    return str(datetime.timedelta(seconds=delta_seconds))