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
def AddMaintenanceReleaseChannel(parser, hidden=False):
    base.ChoiceArgument('--maintenance-release-channel', choices={'week5': 'week5 updates release after the production updates. Use the week5 channel to receive a 5 week advance notification about the upcoming maintenance, so you can prepare your application for the release.', 'production': 'Production updates are stable and recommended for applications in production.', 'preview': 'Preview updates release prior to production updates. You may wish to use the preview channel for dev/test applications so that you can preview their compatibility with your application prior to the production release.'}, help_str="Which channel's updates to apply during the maintenance window. If not specified, Cloud SQL chooses the timing of updates to your instance.", hidden=hidden).AddToParser(parser)