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
def LoadSubElementByPath(self, path):
    """Load a specific sub group or command by path.

    If path is empty, returns the current element.

    Args:
      path: list of str, The names of the elements to load down the hierarchy.

    Returns:
      _CommandCommon, The loaded sub element, or None if it did not exist.
    """
    curr = self
    for part in path:
        curr = curr.LoadSubElement(part)
        if curr is None:
            return None
    return curr