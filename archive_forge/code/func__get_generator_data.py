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
def _get_generator_data(self):
    """Return a dict with data for the sample generator."""
    return {'help': self.help or '', 'dynamic_group_owner': self.dynamic_group_owner, 'driver_option': self.driver_option, 'driver_opts': self._driver_opts}