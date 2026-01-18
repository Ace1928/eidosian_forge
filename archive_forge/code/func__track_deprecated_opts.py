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
def _track_deprecated_opts(self, opt, group=None):
    if hasattr(opt, 'deprecated_opts'):
        for dep_opt in opt.deprecated_opts:
            dep_group = dep_opt.group or 'DEFAULT'
            dep_dest = dep_opt.name
            if dep_dest:
                dep_dest = dep_dest.replace('-', '_')
            if dep_group not in self._deprecated_opts:
                self._deprecated_opts[dep_group] = {dep_dest: {'opt': opt, 'group': group}}
            else:
                self._deprecated_opts[dep_group][dep_dest] = {'opt': opt, 'group': group}