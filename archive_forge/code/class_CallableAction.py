from __future__ import annotations
import argparse
import collections
from functools import update_wrapper
import inspect
import itertools
import operator
import os
import re
import sys
from typing import TYPE_CHECKING
import uuid
import pytest
class CallableAction(argparse.Action):

    def __init__(self, option_strings, dest, default=False, required=False, help=None):
        super().__init__(option_strings=option_strings, dest=dest, nargs=0, const=True, default=default, required=required, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        zeroarg_callback(option_string, values, parser)