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
def _check_required_opts(self, namespace=None):
    """Check that all opts marked as required have values specified.

        :param namespace: the namespace object be checked the required options
        :raises: RequiredOptError
        """
    for info, group in self._all_opt_infos():
        opt = info['opt']
        if opt.required:
            if 'default' in info or 'override' in info:
                continue
            if self._get(opt.dest, group, namespace) is None:
                raise RequiredOptError(opt.name, group)