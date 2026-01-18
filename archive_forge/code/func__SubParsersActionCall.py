from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import os
import re
import sys
import types
import uuid
import argcomplete
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import backend
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import pkg_resources
import six
def _SubParsersActionCall(self, parser, namespace, values, option_string=None):
    """argparse._SubParsersAction.__call__ version 1.2.1 MonkeyPatch."""
    del option_string
    parser_name = values[0]
    arg_strings = values[1:]
    if self.dest is not argparse.SUPPRESS:
        setattr(namespace, self.dest, parser_name)
    try:
        parser = self._name_parser_map[parser_name]
    except KeyError:
        tup = (parser_name, ', '.join(self._name_parser_map))
        msg = argparse._('unknown parser %r (choices: %s)' % tup)
        raise argparse.ArgumentError(self, msg)
    namespace, arg_strings = parser.parse_known_args(arg_strings, namespace)
    if arg_strings:
        vars(namespace).setdefault(argparse._UNRECOGNIZED_ARGS_ATTR, [])
        getattr(namespace, argparse._UNRECOGNIZED_ARGS_ATTR).extend(arg_strings)