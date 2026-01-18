from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.calliope import markdown
from googlecloudsdk.calliope import walker
from googlecloudsdk.core import properties
from googlecloudsdk.core.document_renderers import render_document
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import pkg_resources
import six
class GCloudTreeGenerator(walker.Walker):
    """Generates an external representation of the gcloud CLI tree.

  This implements the resource generator for gcloud meta list-gcloud.
  """

    def Visit(self, node, parent, is_group):
        """Visits each node in the CLI command tree to construct the external rep.

    Args:
      node: group/command CommandCommon info.
      parent: The parent Visit() return value, None at the top level.
      is_group: True if node is a group, otherwise its is a command.

    Returns:
      The subtree parent value, used here to construct an external rep node.
    """
        return cli_tree.Command(node, parent)