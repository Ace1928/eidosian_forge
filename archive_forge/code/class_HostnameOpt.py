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
class HostnameOpt(Opt):
    """Option for a hostname.  Only accepts valid hostnames.

    Option with ``type`` :class:`oslo_config.types.Hostname`

    :param name: the option's name
    :param \\*\\*kwargs: arbitrary keyword arguments passed to :class:`Opt`

    .. versionadded:: 3.8
    """

    def __init__(self, name, **kwargs):
        super(HostnameOpt, self).__init__(name, type=types.Hostname(), **kwargs)