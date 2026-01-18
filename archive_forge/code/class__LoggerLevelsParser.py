from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import getpass
import io
import itertools
import logging
import os
import socket
import struct
import sys
import time
import timeit
import traceback
import types
import warnings
from absl import flags
from absl._collections_abc import abc
from absl.logging import converter
import six
class _LoggerLevelsParser(flags.ArgumentParser):
    """Parser for --logger_levels flag."""

    def parse(self, value):
        if isinstance(value, abc.Mapping):
            return value
        pairs = [pair.strip() for pair in value.split(',') if pair.strip()]
        levels = collections.OrderedDict()
        for name_level in pairs:
            name, level = name_level.split(':', 1)
            name = name.strip()
            level = level.strip()
            levels[name] = level
        return levels