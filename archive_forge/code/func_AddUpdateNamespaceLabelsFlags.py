from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import util as cmd_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def AddUpdateNamespaceLabelsFlags(parser):
    """Adds flags to an argparse parser for updating namespace labels.

  Args:
    parser: The argparse parser to add the flags to.
  """
    _GetUpdateNamespaceLabelsFlag('namespace').AddToParser(parser)
    remove_group = parser.add_mutually_exclusive_group()
    _GetClearNamespaceLabelsFlag('namespace').AddToParser(remove_group)
    _GetRemoveNamespaceLabelsFlag('namespace').AddToParser(remove_group)