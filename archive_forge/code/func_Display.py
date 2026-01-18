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
def Display(self, args, result):
    """Displays the server response to a query.

    This is called higher up the stack to over-write default display behavior.
    What gets displayed depends on the mode in which the query was run.
    'NORMAL': query result rows
    'PLAN': query plan without execution statistics
    'PROFILE': query result rows and the query plan with execution statistics

    Args:
      args: The arguments originally passed to the command.
      result: The output of the command before display.

    Raises:
      ValueError: The query mode is not valid.
    """
    if args.query_mode == 'NORMAL':
        sql.DisplayQueryResults(result, log.out)
    elif args.query_mode == 'PLAN':
        sql.DisplayQueryPlan(result, log.out)
    elif args.query_mode == 'PROFILE':
        if sql.QueryHasAggregateStats(result):
            sql.DisplayQueryAggregateStats(result.stats.queryStats, log.out)
        sql.DisplayQueryPlan(result, log.out)
        sql.DisplayQueryResults(result, log.status)
    else:
        raise ValueError('Invalid query mode: {}'.format(args.query_mode))