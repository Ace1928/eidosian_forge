from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.accesscontextmanager import zones as zones_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.core import resources
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class EnforceAllPerimeterDryRunAlpha(EnforceAllPerimeterDryRun):
    """Enforces the dry-run mode configuration for all Service Perimeters."""
    _API_VERSION = 'v1alpha'