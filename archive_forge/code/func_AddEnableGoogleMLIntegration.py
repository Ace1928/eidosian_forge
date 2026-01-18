from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
def AddEnableGoogleMLIntegration(parser, hidden=False):
    """Adds --enable-google-ml-integration flag."""
    parser.add_argument('--enable-google-ml-integration', required=False, hidden=hidden, help='Enable Vertex AI integration for Google Cloud SQL. Currently, only PostgreSQL is supported.', action=arg_parsers.StoreTrueFalseAction)