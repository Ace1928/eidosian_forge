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
def _get_file_value(self, names, multi=False, normalized=False, current_name=None):
    """Fetch a config file value from the parsed files.

        :param names: a list of (section, name) tuples
        :param multi: a boolean indicating whether to return multiple values
        :param normalized: whether to normalize group names to lowercase
        :param current_name: current name in tuple being checked
        """
    rvalue = []

    def normalize(name):
        if name is None:
            name = 'DEFAULT'
        return _normalize_group_name(name) if normalized else name
    names = [(normalize(section), name) for section, name in names]
    loc = None
    for sections in self._normalized if normalized else self._parsed:
        for section, name in names:
            if section not in sections:
                continue
            if name in sections[section]:
                current_name = current_name or names[0]
                self._check_deprecated((section, name), current_name, names[1:])
                val = sections[section][name]
                if loc is None:
                    loc = LocationInfo(Locations.user, self._sections_to_file.get(section, ''))
                if multi:
                    rvalue = val + rvalue
                else:
                    return (val, loc)
    if multi and rvalue != []:
        return (rvalue, loc)
    raise KeyError