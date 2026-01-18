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
def _GetFlagsHelper(arg, level=0, required=True):
    """GetFlags() helper that adds to flags."""
    if arg.is_hidden:
        return
    if arg.is_group:
        if level and required:
            required = arg.is_required
        for arg in arg.arguments:
            _GetFlagsHelper(arg, level=level + 1, required=required)
    else:
        show_inverted = getattr(arg, 'show_inverted', None)
        if show_inverted:
            arg = show_inverted
        if arg.option_strings and (not arg.is_positional) and (not arg.is_global) and (not optional or not required or (not arg.is_required)):
            flags.add(sorted(arg.option_strings)[0])