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
def PluralMembershipsResourceNames(args):
    """Gets a list of membership resource names from a --memberships resource arg.

  Args:
    args: arguments provided to a command, including a plural memberships
      resource arg

  Returns:
    A list of membership resource names (e.g.
    projects/x/locations/y/memberships/z)
  """
    return [m.RelativeName() for m in args.CONCEPTS.memberships.Parse()]