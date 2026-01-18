from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class BackendBindings(base.Group):
    """Create and manage backend bindings with Compute Engine backend service for KubeRun services."""