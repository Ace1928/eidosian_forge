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
def _get_argparse_container(self, parser, group):
    """Returns an argparse._ArgumentGroup.

        :param parser: an argparse.ArgumentParser
        :param group: an (optional) OptGroup object
        :returns: an argparse._ArgumentGroup if group is given, else parser
        """
    if group is not None:
        return group._get_argparse_group(parser)
    else:
        return parser