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
def LoadAllSubElements(self, recursive=False, ignore_load_errors=False):
    """Load all the sub groups and commands of this group.

    Args:
      recursive: bool, True to continue loading all sub groups, False, to just
        load the elements under the group.
      ignore_load_errors: bool, True to ignore command load failures. This
        should only be used when it is not critical that all data is returned,
        like for optimizations like static tab completion.

    Returns:
      int, The total number of elements loaded.
    """
    total = 0
    for name in self.AllSubElements():
        try:
            element = self.LoadSubElement(name)
            total += 1
        except:
            element = None
            if not ignore_load_errors:
                raise
        if element and recursive:
            total += element.LoadAllSubElements(recursive=recursive, ignore_load_errors=ignore_load_errors)
    return total