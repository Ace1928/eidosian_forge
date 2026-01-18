from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.spanner import database_sessions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.spanner import resource_args
from googlecloudsdk.command_lib.spanner import sql
from googlecloudsdk.command_lib.spanner.sql import QueryHasDml
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateSession(args, database_role=None):
    """Creates a session.

  Args:
    args: an argparse namespace. All the arguments that were provided to the
      command invocation.
    database_role: Cloud Spanner database role which owns this session.

  Returns:
    A session reference to be used to execute the sql.
  """
    session_name = database_sessions.Create(args.CONCEPTS.database.Parse(), database_role)
    return resources.REGISTRY.ParseRelativeName(relative_name=session_name.name, collection='spanner.projects.instances.databases.sessions')