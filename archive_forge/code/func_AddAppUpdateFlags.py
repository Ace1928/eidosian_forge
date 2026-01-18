from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app.api import appengine_app_update_api_client
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import log
from googlecloudsdk.core.console import progress_tracker
def AddAppUpdateFlags(parser):
    """Add the common flags to a app update command."""
    parser.add_argument('--split-health-checks', action=arg_parsers.StoreTrueFalseAction, help='Enables/disables split health checks by default on new deployments.')
    parser.add_argument('--service-account', help='The app-level default service account to update the app with.')