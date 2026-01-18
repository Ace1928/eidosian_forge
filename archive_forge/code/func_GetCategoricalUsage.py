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
def GetCategoricalUsage(command, categories):
    """Constructs an alternative Usage markdown string organized into categories.

  The string is formatted as a series of tables; first, there's a table for
  each category of subgroups, next, there's a table for each category of
  subcommands. Each table element is printed under the category defined in the
  surface definition of the command or group with a short summary describing its
  functionality. In either set of tables (groups or commands), if there are no
  categories to display, there will be only be one table listing elements
  lexicographically. If both the sets of tables (groups and commands) have no
  categories to display, then an empty string is returned.

  Args:
    command: calliope._CommandCommon, The command object that we're helping.
    categories: A dictionary mapping category name to the set of elements
      belonging to that category.

  Returns:
    str, The command usage markdown string organized into categories.
  """
    command_key = 'command'
    command_group_key = 'command_group'

    def _WriteTypeUsageTextToBuffer(buf, categories, key_name):
        """Writes the markdown string to the buffer passed by reference."""
        single_category_is_other = False
        if len(categories[key_name]) == 1 and base.UNCATEGORIZED_CATEGORY in categories[key_name]:
            single_category_is_other = True
        buf.write('\n\n')
        buf.write('# Available {type}s for {group}:\n'.format(type=' '.join(key_name.split('_')), group=' '.join(command.GetPath())))
        for category, elements in sorted(six.iteritems(categories[key_name])):
            if not single_category_is_other:
                buf.write('\n### {category}\n\n'.format(category=category))
            buf.write('---------------------- | ---\n')
            for element in sorted(elements, key=lambda e: e.name):
                short_help = None
                if element.name == 'alpha':
                    short_help = element.short_help[10:]
                elif element.name == 'beta':
                    short_help = element.short_help[9:]
                else:
                    short_help = element.short_help
                buf.write('{name} | {description}\n'.format(name=element.name.replace('_', '-'), description=short_help))

    def _ShouldCategorize(categories):
        """Ensures the categorization has real categories and is not just all Uncategorized."""
        if not categories[command_key].keys() and (not categories[command_group_key].keys()):
            return False
        if set(list(categories[command_key].keys()) + list(categories[command_group_key].keys())) == set([base.UNCATEGORIZED_CATEGORY]):
            return False
        return True
    if not _ShouldCategorize(categories):
        return ''
    buf = io.StringIO()
    if command_group_key in categories:
        _WriteTypeUsageTextToBuffer(buf, categories, command_group_key)
    if command_key in categories:
        _WriteTypeUsageTextToBuffer(buf, categories, command_key)
    return buf.getvalue()