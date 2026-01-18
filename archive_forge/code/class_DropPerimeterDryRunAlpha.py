from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.accesscontextmanager import zones as zones_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.accesscontextmanager import perimeters
from googlecloudsdk.command_lib.accesscontextmanager import policies
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class DropPerimeterDryRunAlpha(DropPerimeterDryRun):
    """Resets the dry-run mode configuration of a Service Perimeter."""
    _API_VERSION = 'v1alpha'