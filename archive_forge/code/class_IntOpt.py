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
class IntOpt(Opt):
    """Option with Integer type

    Option with ``type`` :class:`oslo_config.types.Integer`

    :param name: the option's name
    :param min: minimum value the integer can take
    :param max: maximum value the integer can take
    :param choices: Optional sequence of either valid values or tuples of valid
        values with descriptions.
    :param \\*\\*kwargs: arbitrary keyword arguments passed to :class:`Opt`

    .. versionchanged:: 1.15

       Added *min* and *max* parameters.


    .. versionchanged:: 9.3.0

       Added *choices* parameter.
    """

    def __init__(self, name, min=None, max=None, choices=None, **kwargs):
        super(IntOpt, self).__init__(name, type=types.Integer(min=min, max=max, choices=choices), **kwargs)