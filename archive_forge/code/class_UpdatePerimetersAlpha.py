from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.accesscontextmanager import zones as zones_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.accesscontextmanager import perimeters
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.args import repeated
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class UpdatePerimetersAlpha(UpdatePerimetersGA):
    """Update an existing access zone."""
    _INCLUDE_UNRESTRICTED = False
    _API_VERSION = 'v1alpha'

    @staticmethod
    def Args(parser):
        UpdatePerimetersGA.ArgsVersioned(parser, version='v1alpha')