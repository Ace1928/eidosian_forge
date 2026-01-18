import argparse
import contextlib
import io
import json
import logging
import os
import pkgutil
import sys
from apitools.base.py import exceptions
from apitools.gen import gen_client_lib
from apitools.gen import util
class _SplitCommaSeparatedList(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.split(','))