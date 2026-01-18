from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import logs as cb_logs
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.builds import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class LogAlpha(LogBeta):
    """Stream the logs for a build."""
    _support_gcl = True