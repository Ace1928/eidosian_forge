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
def _validate_cli_options(self, namespace):
    for opt, group in sorted(self._all_cli_opts(), key=lambda x: x[0].name):
        group_name = group.name if group else None
        try:
            value, loc = opt._get_from_namespace(namespace, group_name)
        except KeyError:
            continue
        value = self._substitute(value, group=group, namespace=namespace)
        try:
            self._convert_value(value, opt)
        except ValueError:
            sys.stderr.write('argument --%s: Invalid %s value: %s\n' % (opt.dest, repr(opt.type), value))
            raise SystemExit