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
def RunGroupFilter(self, context, args):
    """Constructs and runs the Filter() method of all parent groups.

    This recurses up to the root group and then constructs each group and runs
    its Filter() method down the tree.

    Args:
      context: {}, The context dictionary that Filter() can modify.
      args: The argparse namespace.
    """
    if self._parent_group:
        self._parent_group.RunGroupFilter(context, args)
    self._common_type().Filter(context, args)