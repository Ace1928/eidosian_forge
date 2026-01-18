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
def _add_parsed_config_file(self, filename, sections, normalized):
    """Add a parsed config file to the list of parsed files.

        :param filename: the full name of the file that was parsed
        :param sections: a mapping of section name to dicts of config values
        :param normalized: sections mapping with section names normalized
        :raises: ConfigFileValueError
        """
    for s in sections:
        self._sections_to_file[s] = filename
    self._parsed.insert(0, sections)
    self._normalized.insert(0, normalized)