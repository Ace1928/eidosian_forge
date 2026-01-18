import sys
import os
import re
import copy
import warnings
import subprocess
import textwrap
from glob import glob
from functools import reduce
from configparser import NoOptionError
from configparser import RawConfigParser as ConfigParser
from distutils.errors import DistutilsError
from distutils.dist import Distribution
import sysconfig
from numpy.distutils import log
from distutils.util import get_platform
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import (is_sequence, is_string,
from numpy.distutils.command.config import config as cmd_config
from numpy.distutils import customized_ccompiler as _customized_ccompiler
from numpy.distutils import _shell_utils
import distutils.ccompiler
import tempfile
import shutil
import platform
def _parse_env_order(base_order, env):
    """ Parse an environment variable `env` by splitting with "," and only returning elements from `base_order`

    This method will sequence the environment variable and check for their
    individual elements in `base_order`.

    The items in the environment variable may be negated via '^item' or '!itema,itemb'.
    It must start with ^/! to negate all options.

    Raises
    ------
    ValueError: for mixed negated and non-negated orders or multiple negated orders

    Parameters
    ----------
    base_order : list of str
       the base list of orders
    env : str
       the environment variable to be parsed, if none is found, `base_order` is returned

    Returns
    -------
    allow_order : list of str
        allowed orders in lower-case
    unknown_order : list of str
        for values not overlapping with `base_order`
    """
    order_str = os.environ.get(env, None)
    base_order = [order.lower() for order in base_order]
    if order_str is None:
        return (base_order, [])
    neg = order_str.startswith('^') or order_str.startswith('!')
    order_str_l = list(order_str)
    sum_neg = order_str_l.count('^') + order_str_l.count('!')
    if neg:
        if sum_neg > 1:
            raise ValueError(f"Environment variable '{env}' may only contain a single (prefixed) negation: {order_str}")
        order_str = order_str[1:]
    elif sum_neg > 0:
        raise ValueError(f"Environment variable '{env}' may not mix negated an non-negated items: {order_str}")
    orders = order_str.lower().split(',')
    unknown_order = []
    if neg:
        allow_order = base_order.copy()
        for order in orders:
            if not order:
                continue
            if order not in base_order:
                unknown_order.append(order)
                continue
            if order in allow_order:
                allow_order.remove(order)
    else:
        allow_order = []
        for order in orders:
            if not order:
                continue
            if order not in base_order:
                unknown_order.append(order)
                continue
            if order not in allow_order:
                allow_order.append(order)
    return (allow_order, unknown_order)