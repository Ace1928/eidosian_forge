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
def _find_deprecated_opts(self, opt_name, group=None):
    real_opt_name = None
    real_group_name = None
    group_name = group or 'DEFAULT'
    if hasattr(group_name, 'name'):
        group_name = group_name.name
    dep_group = self._deprecated_opts.get(group_name)
    if dep_group:
        real_opt_dict = dep_group.get(opt_name)
        if real_opt_dict:
            real_opt_name = real_opt_dict['opt'].name
            if real_opt_dict['group']:
                real_group_name = real_opt_dict['group'].name
    return (real_opt_name, real_group_name)