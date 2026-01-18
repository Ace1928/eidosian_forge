import argparse
import collections
from collections import abc
import copy
import enum
import errno
import functools
import glob
import inspect
import itertools
import logging
import os
import string
import sys
from oslo_config import iniparser
from oslo_config import sources
import oslo_config.sources._environment as _environment
from oslo_config import types
import stevedore
class ConfigDirNotFoundError(Error):
    """Raised if the requested config-dir is not found."""

    def __init__(self, config_dir):
        self.config_dir = config_dir

    def __str__(self):
        return 'Failed to read config file directory: %s' % self.config_dir