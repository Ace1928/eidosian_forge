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
def GetArgUsage(arg, brief=False, definition=False, markdown=False, optional=True, top=False, remainder_usage=None, value=True, hidden=False):
    """Returns the argument usage string for arg or all nested groups in arg.

  Mutually exclusive args names are separated by ' | ', otherwise ' '.
  Required groups are enclosed in '(...)', otherwise '[...]'. Required args
  in a group are separated from the optional args by ' : '.

  Args:
    arg: The argument to get usage from.
    brief: bool, If True, only display one version of a flag that has
        multiple versions, and do not display the default value.
    definition: bool, Definition list usage if True.
    markdown: bool, Add markdown if True.
    optional: bool, Include optional flags if True.
    top: bool, True if args is the top level group.
    remainder_usage: [str], Append REMAINDER usage here instead of the return.
    value: bool, If true display flag name=value for non-Boolean flags.
    hidden: bool, Include hidden args if True.

  Returns:
    The argument usage string for arg or all nested groups in arg.
  """
    if arg.is_hidden and (not hidden):
        return ''
    if arg.is_group:
        singleton = GetSingleton(arg)
        if singleton and (singleton.is_group or singleton.nargs != argparse.REMAINDER):
            arg = singleton
    if not arg.is_group:
        if arg.is_positional:
            usage = GetPositionalUsage(arg, markdown=markdown)
        else:
            if isinstance(arg, arg_parsers.StoreTrueFalseAction):
                inverted = InvertedValue.BOTH
            elif not definition and getattr(arg, 'inverted_synopsis', False):
                inverted = InvertedValue.INVERTED
            else:
                inverted = InvertedValue.NORMAL
            usage = GetFlagUsage(arg, brief=brief, markdown=markdown, inverted=inverted, value=value)
        if usage and top and (not arg.is_required):
            usage = _MarkOptional(usage)
        return usage
    sep = ' | ' if arg.is_mutex else ' '
    positional_args = []
    required_usage = []
    optional_usage = []
    if remainder_usage is None:
        include_remainder_usage = True
        remainder_usage = []
    else:
        include_remainder_usage = False
    arguments = sorted(arg.arguments, key=GetArgSortKey) if arg.sort_args else arg.arguments
    for a in arguments:
        if a.is_hidden and (not hidden):
            continue
        if a.is_group:
            singleton = GetSingleton(a)
            if singleton:
                a = singleton
        if not a.is_group and a.nargs == argparse.REMAINDER:
            remainder_usage.append(GetArgUsage(a, markdown=markdown, value=value, hidden=hidden))
        elif _IsPositional(a):
            positional_args.append(a)
        else:
            usage = GetArgUsage(a, markdown=markdown, value=value, hidden=hidden)
            if not usage:
                continue
            if a.is_required:
                if usage not in required_usage:
                    required_usage.append(usage)
            else:
                if top:
                    usage = _MarkOptional(usage)
                if usage not in optional_usage:
                    optional_usage.append(usage)
    positional_usage = []
    all_other_usage = []
    nesting = 0
    optional_positionals = False
    if positional_args:
        nesting = 0
        for a in positional_args:
            usage = GetArgUsage(a, markdown=markdown, hidden=hidden)
            if not usage:
                continue
            if not a.is_required:
                optional_positionals = True
                usage_orig = usage
                usage = _MarkOptional(usage)
                if usage != usage_orig:
                    nesting += 1
            positional_usage.append(usage)
        if nesting:
            positional_usage[-1] = '{}{}'.format(positional_usage[-1], ']' * nesting)
    if required_usage:
        all_other_usage.append(sep.join(required_usage))
    if optional_usage:
        if optional:
            if not top and (positional_args and (not optional_positionals) or required_usage):
                all_other_usage.append(':')
            all_other_usage.append(sep.join(optional_usage))
        elif brief and top:
            all_other_usage.append('[optional flags]')
    if brief:
        all_usage = positional_usage + sorted(all_other_usage, key=_GetArgUsageSortKey)
    else:
        all_usage = positional_usage + all_other_usage
    if remainder_usage and include_remainder_usage:
        all_usage.append(' '.join(remainder_usage))
    usage = ' '.join(all_usage)
    if arg.is_required:
        return '({})'.format(usage)
    if not top and len(all_usage) > 1:
        usage = _MarkOptional(usage)
    return usage