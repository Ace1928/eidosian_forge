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