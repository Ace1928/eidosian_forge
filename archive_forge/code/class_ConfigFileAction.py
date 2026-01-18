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
class ConfigFileAction(argparse.Action):
    """An argparse action for --config-file.

        As each --config-file option is encountered, this action adds the
        value to the config_file attribute on the _Namespace object but also
        parses the configuration file and stores the values found also in
        the _Namespace object.
        """

    def __call__(self, parser, namespace, values, option_string=None):
        """Handle a --config-file command line argument.

            :raises: ConfigFileParseError, ConfigFileValueError
            """
        if getattr(namespace, self.dest, None) is None:
            setattr(namespace, self.dest, [])
        items = getattr(namespace, self.dest)
        items.append(values)
        ConfigParser._parse_file(values, namespace)