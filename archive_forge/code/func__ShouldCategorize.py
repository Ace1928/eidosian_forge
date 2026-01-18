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
def _ShouldCategorize(categories):
    """Ensures the categorization has real categories and is not just all Uncategorized."""
    if not categories[command_key].keys() and (not categories[command_group_key].keys()):
        return False
    if set(list(categories[command_key].keys()) + list(categories[command_group_key].keys())) == set([base.UNCATEGORIZED_CATEGORY]):
        return False
    return True