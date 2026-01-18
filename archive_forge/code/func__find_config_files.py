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
def _find_config_files(project, prog, extension):
    if prog is None:
        prog = os.path.basename(sys.argv[0])
        if prog.endswith('.py'):
            prog = prog[:-3]
    cfg_dirs = _get_config_dirs(project)
    config_files = (_search_dirs(cfg_dirs, p, extension) for p in [project, prog] if p)
    return [x for x in config_files if x]