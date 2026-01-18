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
def _save_driver_opts(self, opts):
    """Save known driver opts.

        :param opts: mapping between driver name and list of opts
        :type opts: dict

        """
    self._driver_opts.update(opts)