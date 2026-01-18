from __future__ import division
from __future__ import print_function
import collections
import contextlib
import errno
import functools
import os
import socket
import stat
import sys
import threading
import warnings
from collections import namedtuple
from socket import AF_INET
from socket import SOCK_DGRAM
from socket import SOCK_STREAM
def _remove_dead_reminders(self, input_dict, name):
    """In case the number of keys changed between calls (e.g. a
        disk disappears) this removes the entry from self.reminders.
        """
    old_dict = self.cache[name]
    gone_keys = set(old_dict.keys()) - set(input_dict.keys())
    for gone_key in gone_keys:
        for remkey in self.reminder_keys[name][gone_key]:
            del self.reminders[name][remkey]
        del self.reminder_keys[name][gone_key]