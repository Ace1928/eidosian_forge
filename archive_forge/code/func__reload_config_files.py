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
def _reload_config_files(self):
    namespace = self._parse_config_files()
    if namespace._files_not_found:
        raise ConfigFilesNotFoundError(namespace._files_not_found)
    if namespace._files_permission_denied:
        raise ConfigFilesPermissionDeniedError(namespace._files_permission_denied)
    self._check_required_opts(namespace)
    return namespace