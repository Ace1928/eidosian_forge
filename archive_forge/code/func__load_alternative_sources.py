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
def _load_alternative_sources(self):
    for source_group_name in self.config_source:
        source = self._open_source_from_opt_group(source_group_name)
        if source is not None:
            self._sources.append(source)