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