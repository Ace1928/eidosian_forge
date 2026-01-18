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
def _get_cli_value(self, names, positional=False):
    """Fetch a CLI option value.

        Look up the value of a CLI option. The value itself may have come from
        parsing the command line or parsing config files specified on the
        command line. Type conversion have already been performed for CLI
        options at this point.

        :param names: a list of (section, name) tuples
        :param positional: whether this is a positional option
        """
    for group_name, name in names:
        name = name if group_name is None else group_name + '_' + name
        value = getattr(self, name, None)
        if value is not None:
            if positional and (not value):
                continue
            return value
    raise KeyError