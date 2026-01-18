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
@classmethod
def _list_options_for_discovery(cls, default_config_files, default_config_dirs):
    """Return options to be used by list_opts() for the sample generator."""
    options = cls._make_config_options(default_config_files, default_config_dirs)
    options.append(cls._config_source_opt)
    return options