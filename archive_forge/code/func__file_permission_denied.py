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
def _file_permission_denied(self, config_file):
    """Record that we have no permission to open a config file.

        :param config_file: the path to the failed file
        """
    self._files_permission_denied.append(config_file)