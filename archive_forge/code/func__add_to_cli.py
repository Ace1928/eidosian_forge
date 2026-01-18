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
def _add_to_cli(self, parser, group=None):
    """Add argparse sub-parsers and invoke the handler method."""
    dest = self.dest
    if group is not None:
        dest = group.name + '_' + dest
    subparsers = parser.add_subparsers(dest=dest, title=self.title, description=self.description, help=self.help)
    subparsers.required = True
    if self.handler is not None:
        self.handler(subparsers)