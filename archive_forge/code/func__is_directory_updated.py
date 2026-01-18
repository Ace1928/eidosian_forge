import collections.abc
import copy
import logging
import os
import typing as ty
import warnings
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import strutils
import yaml
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy._i18n import _
from oslo_policy import _parser
from oslo_policy import opts
@staticmethod
def _is_directory_updated(cache, path):
    mtime = 0
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise ValueError('{} is not a directory'.format(path))
        files = [path] + [os.path.join(path, file) for file in os.listdir(path)]
        mtime = os.path.getmtime(max(files, key=os.path.getmtime))
    cache_info = cache.setdefault(path, {})
    if mtime > cache_info.get('mtime', 0):
        cache_info['mtime'] = mtime
        return True
    return False