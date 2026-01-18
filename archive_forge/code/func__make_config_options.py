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
@staticmethod
def _make_config_options(default_config_files, default_config_dirs):
    return [_ConfigFileOpt('config-file', default=default_config_files, metavar='PATH', help='Path to a config file to use. Multiple config files can be specified, with values in later files taking precedence. Defaults to %(default)s. This option must be set from the command-line.'), _ConfigDirOpt('config-dir', metavar='DIR', default=default_config_dirs, help='Path to a config directory to pull `*.conf` files from. This file set is sorted, so as to provide a predictable parse order if individual options are over-ridden. The set is parsed after the file(s) specified via previous --config-file, arguments hence over-ridden options in the directory take precedence. This option must be set from the command-line.')]