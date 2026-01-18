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
class _CommandTimer(object):
    """A class for timing the execution of a command."""

    def __init__(self):
        self.__start = 0
        self.__events = []
        self.__total_rpc_duration = 0
        self.__total_local_duration = 0
        self.__category = 'unknown'
        self.__action = 'unknown'
        self.__label = None
        self.__flag_names = None

    def SetContext(self, category, action, label, flag_names):
        self.__category = category
        self.__action = action
        self.__label = label
        self.__flag_names = flag_names

    def GetContext(self):
        return (self.__category, self.__action, self.__label, self.__flag_names)

    def Event(self, name, event_time=None):
        time_millis = GetTimeMillis(event_time)
        if name is _START_EVENT:
            self.__start = time_millis
            return
        self.__events.append(_TimedEvent(name, time_millis))
        if name is _TOTAL_EVENT:
            self.__total_local_duration = time_millis - self.__start
            self.__total_local_duration -= self.__total_rpc_duration

    def AddRPCDuration(self, duration_in_ms):
        self.__total_rpc_duration += duration_in_ms

    def GetTimings(self):
        """Returns the timings for the recorded events."""
        timings = []
        for event in self.__events:
            timings.append((event.name, event.time_millis - self.__start))
        timings.extend([(_LOCAL_EVENT, self.__total_local_duration), (_REMOTE_EVENT, self.__total_rpc_duration)])
        return timings