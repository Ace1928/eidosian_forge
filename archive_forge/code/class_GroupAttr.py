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
class GroupAttr(abc.Mapping):
    """Helper class.

        Represents the option values of a group as a mapping and attributes.
        """

    def __init__(self, conf, group):
        """Construct a GroupAttr object.

            :param conf: a ConfigOpts object
            :param group: an OptGroup object
            """
        self._conf = conf
        self._group = group

    def __getattr__(self, name):
        """Look up an option value and perform template substitution."""
        return self._conf._get(name, self._group)

    def __getitem__(self, key):
        """Look up an option value and perform string substitution."""
        return self.__getattr__(key)

    def __contains__(self, key):
        """Return True if key is the name of a registered opt or group."""
        return key in self._group._opts

    def __iter__(self):
        """Iterate over all registered opt and group names."""
        for key in self._group._opts.keys():
            yield key

    def __len__(self):
        """Return the number of options and option groups."""
        return len(self._group._opts)