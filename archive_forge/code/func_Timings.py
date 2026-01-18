from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import json
import os
import pickle
import platform
import socket
import subprocess
import sys
import tempfile
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
import six
import six.moves.urllib.error
import six.moves.urllib.parse
import six.moves.urllib.request
def Timings(self, timer):
    """Extracts relevant data from timer."""
    total_latency = None
    timings = timer.GetTimings()
    sub_event_latencies = []
    for timing in timings:
        if not total_latency and timing[0] == _TOTAL_EVENT:
            total_latency = timing[1]
        sub_event_latencies.append({'key': timing[0], 'latency_ms': timing[1]})
    return (total_latency, sub_event_latencies)