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
def _parse_cli_opts(self, args):
    """Parse command line options.

        Initializes the command line option parser and parses the supplied
        command line arguments.

        :param args: the command line arguments
        :returns: a _Namespace object containing the parsed option values
        :raises: SystemExit, DuplicateOptError
                 ConfigFileParseError, ConfigFileValueError

        """
    self._args = args
    for opt, group in self._all_cli_opts():
        opt._add_to_cli(self._oparser, group)
    return self._parse_config_files()