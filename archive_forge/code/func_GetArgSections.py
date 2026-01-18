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
def GetArgSections(arguments, is_root, is_group, sort_top_level_args):
    """Returns the positional/flag sections in document order.

  Args:
    arguments: [Flag|Positional], The list of arguments for this command or
      group.
    is_root: bool, True if arguments are for the CLI root command.
    is_group: bool, True if arguments are for a command group.
    sort_top_level_args: bool, True if top level arguments should be sorted.

  Returns:
    ([Section] global_flags)
      global_flags - The sorted list of global flags if command is not the root.
  """
    categories = collections.OrderedDict()
    dests = set()
    global_flags = set()
    if not is_root and is_group:
        global_flags = {'--help'}
    for arg in arguments:
        if arg.is_hidden:
            continue
        if _IsPositional(arg):
            category = 'POSITIONAL ARGUMENTS'
            if category not in categories:
                categories[category] = []
            categories[category].append(arg)
            continue
        if arg.is_global and (not is_root):
            for a in arg.arguments if arg.is_group else [arg]:
                if a.option_strings and (not a.is_hidden):
                    flag = a.option_strings[0]
                    if not is_group and flag.startswith('--'):
                        global_flags.add(flag)
            continue
        if arg.is_required:
            category = 'REQUIRED'
        else:
            category = getattr(arg, 'category', None) or 'OTHER'
        if hasattr(arg, 'dest'):
            if arg.dest in dests:
                continue
            dests.add(arg.dest)
        if category not in categories:
            categories[category] = []
        categories[category].append(arg)
    sections = []
    if is_root:
        common = 'GLOBAL'
    else:
        common = base.COMMONLY_USED_FLAGS
    if sort_top_level_args:
        initial_categories = ['POSITIONAL ARGUMENTS', 'REQUIRED', common, 'OTHER']
        remaining_categories = sorted([c for c in categories if c not in initial_categories])
    else:
        initial_categories = ['POSITIONAL ARGUMENTS', 'REQUIRED', common]
        remaining_categories = [c for c in categories if c not in initial_categories]

    def _GetArgHeading(category):
        """Returns the arg section heading for an arg category."""
        if category == 'OTHER':
            if set(remaining_categories) - set(['OTHER']):
                other_flags_heading = 'FLAGS'
            elif common in categories:
                other_flags_heading = 'OTHER FLAGS'
            elif 'REQUIRED' in categories:
                other_flags_heading = 'OPTIONAL FLAGS'
            else:
                other_flags_heading = 'FLAGS'
            return other_flags_heading
        if 'ARGUMENTS' in category or 'FLAGS' in category:
            return category
        return category + ' FLAGS'
    for category in initial_categories + remaining_categories:
        if category not in categories:
            continue
        sections.append(Section(_GetArgHeading(category), ArgumentWrapper(arguments=categories[category], sort_args=sort_top_level_args)))
    return (sections, global_flags)