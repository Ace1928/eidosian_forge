from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class Core(base.Group):
    """Top level command to interact with KubeRun resources on Google Kubernetes Engine clusters.

  Use this set of commands to create and manage KubeRun resources
  like services, revisions, events, and triggers.
  """
    detailed_help = {'EXAMPLES': '          To list your KubeRun services, run:\n\n            $ {command} services list\n      '}