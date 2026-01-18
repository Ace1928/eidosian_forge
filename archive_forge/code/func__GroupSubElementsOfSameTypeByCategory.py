from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import re
import textwrap
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.calliope import display
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.calliope.concepts import handlers
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.util import text
import six
def _GroupSubElementsOfSameTypeByCategory(elements):
    """Returns dictionary mapping specific to element type."""
    categorized_dict = collections.defaultdict(set)
    for element in elements.values():
        if not element.IsHidden():
            if element.category:
                categorized_dict[element.category].add(element)
            else:
                categorized_dict[base.UNCATEGORIZED_CATEGORY].add(element)
    return categorized_dict