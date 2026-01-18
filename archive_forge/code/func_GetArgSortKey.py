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
def GetArgSortKey(arg):
    """Arg key function for sorted."""
    name = re.sub(' +', ' ', re.sub('[](){}|[]', '', GetArgUsage(arg, value=False, hidden=True) or ''))
    if arg.is_group:
        singleton = GetSingleton(arg)
        if singleton:
            arg = singleton
    if arg.is_group:
        if _IsPositional(arg):
            return (1, '')
        if arg.is_required:
            return (6, name)
        return (7, name)
    elif arg.nargs == argparse.REMAINDER:
        return (8, name)
    if arg.is_positional:
        return (1, '')
    if arg.is_required:
        return (2, name)
    return _GetArgUsageSortKey(name)