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
def _add_to_argparse(self, parser, container, name, short, kwargs, prefix='', positional=False, deprecated_names=None):
    """Add an option to an argparse parser or group.

        :param container: an argparse._ArgumentGroup object
        :param name: the opt name
        :param short: the short opt name
        :param kwargs: the keyword arguments for add_argument()
        :param prefix: an optional prefix to prepend to the opt name
        :param positional: whether the option is a positional CLI argument
        :param deprecated_names: list of deprecated option names
        """

    def hyphen(arg):
        return arg if not positional else ''
    if positional:
        prefix = prefix.replace('-', '_')
        name = name.replace('-', '_')
    args = [hyphen('--') + prefix + name]
    if short:
        args.append(hyphen('-') + short)
    for deprecated_name in deprecated_names:
        args.append(hyphen('--') + deprecated_name)
    parser.add_parser_argument(container, *args, **kwargs)