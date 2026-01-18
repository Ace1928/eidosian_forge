from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import copy
import difflib
import enum
import io
import re
import sys
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import arg_parsers_usage_text
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import util as format_util
import six
def GetSingleton(args):
    """Returns the single non-hidden arg in args.arguments or None."""
    singleton = None
    for arg in args.arguments:
        if arg.is_hidden:
            continue
        if arg.is_group:
            arg = GetSingleton(arg)
            if not arg:
                return None
        if singleton:
            return None
        singleton = arg
    if singleton and (not isinstance(args, ArgumentWrapper)) and (singleton.is_required != args.is_required):
        singleton = copy.copy(singleton)
        singleton.is_required = args.is_required
    return singleton