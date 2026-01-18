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
def _get_deprecated_cli_name(self, dname, dgroup, prefix=''):
    """Build a CLi arg name for deprecated options.

        Either a deprecated name or a deprecated group or both or
        neither can be supplied:

          dname, dgroup -> dgroup + '-' + dname
          dname         -> dname
          dgroup        -> dgroup + '-' + self.name
          neither        -> None

        :param dname: a deprecated name, which can be None
        :param dgroup: a deprecated group, which can be None
        :param prefix: an prefix to append to (for example 'no' or '')
        :returns: a CLI argument name
        """
    if dgroup == 'DEFAULT':
        dgroup = None
    if dname is None and dgroup is None:
        return None
    if dname is None:
        dname = self.name
    return self._get_argparse_prefix(prefix, dgroup) + dname