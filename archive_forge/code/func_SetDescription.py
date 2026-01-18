from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def SetDescription(args, workflow, updated_fields):
    """Set description for the workflow based on the arguments.

  Also update updated_fields accordingly.

  Args:
    args: Args passed to the command.
    workflow: The workflow in which to set the description.
    updated_fields: A list to which a description field will be added if needed.
  """
    if args.description is not None:
        workflow.description = args.description
        updated_fields.append('description')