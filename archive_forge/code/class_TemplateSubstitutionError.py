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
class TemplateSubstitutionError(Error):
    """Raised if an error occurs substituting a variable in an opt value."""

    def __str__(self):
        return 'template substitution error: %s' % self.msg