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
def GetMembershipProjects(memberships):
    """Returns all unique project identifiers of the given membership names.

  ListMemberships should use the same identifier (all number or all ID) in
  membership names. Users can convert their own project identifiers for manually
  entering arguments.

  Args:
    memberships: A list of full membership resource names

  Returns:
    A list of project identifiers in the parents of the memberships

  Raises: googlecloudsdk.core.exceptions.Error if unable to parse any membership
  name
  """
    projects = set()
    for m in memberships:
        match = re.match('projects\\/(.*)\\/locations\\/(.*)\\/memberships\\/(.*)', m)
        if not match:
            raise exceptions.Error('Unable to parse membership {} (expected projects/*/locations/*/memberships/*)'.format(m))
        projects.add(match.group(1))
    return list(projects)