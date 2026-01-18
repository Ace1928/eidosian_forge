import sys
import os
import io
import time
import re
import types
from typing import Protocol
import zipfile
import zipimport
import warnings
import stat
import functools
import pkgutil
import operator
import platform
import collections
import plistlib
import email.parser
import errno
import tempfile
import textwrap
import inspect
import ntpath
import posixpath
import importlib
import importlib.machinery
from pkgutil import get_importer
import _imp
from os import utime
from os import open as os_open
from os.path import isdir, split
from pkg_resources.extern.jaraco.text import (
from pkg_resources.extern import platformdirs
from pkg_resources.extern import packaging
def _resolve_dist(self, req, best, replace_conflicting, env, installer, required_by, to_activate):
    dist = best.get(req.key)
    if dist is None:
        dist = self.by_key.get(req.key)
        if dist is None or (dist not in req and replace_conflicting):
            ws = self
            if env is None:
                if dist is None:
                    env = Environment(self.entries)
                else:
                    env = Environment([])
                    ws = WorkingSet([])
            dist = best[req.key] = env.best_match(req, ws, installer, replace_conflicting=replace_conflicting)
            if dist is None:
                requirers = required_by.get(req, None)
                raise DistributionNotFound(req, requirers)
        to_activate.append(dist)
    if dist not in req:
        dependent_req = required_by[req]
        raise VersionConflict(dist, req).with_context(dependent_req)
    return dist