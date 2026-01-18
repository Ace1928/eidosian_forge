from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import atexit
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import exceptions
from googlecloudsdk.api_lib.sql import instances as instances_api_util
from googlecloudsdk.api_lib.sql import network
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.sql import flags as sql_flags
from googlecloudsdk.command_lib.sql import instances as instances_command_util
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import text
import six
import six.moves.http_client
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class ConnectBeta(base.Command):
    """Connects to a Cloud SQL instance."""
    detailed_help = DETAILED_ALPHA_BETA_HELP

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command."""
        AddBaseArgs(parser)
        AddBetaArgs(parser)
        sql_flags.AddDatabase(parser, 'The PostgreSQL or SQL Server database to connect to.')

    def Run(self, args):
        """Connects to a Cloud SQL instance."""
        return RunProxyConnectCommand(args, supports_database=True)