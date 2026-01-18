from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.accesscontextmanager import authorized_orgs as authorized_orgs_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.accesscontextmanager import authorized_orgs
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.args import repeated
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class UpdateAuthorizedOrgsDescsAlpha(UpdateAuthorizedOrgsDescsBase):
    """Update an existing authorized orgsd desc."""
    _INCLUDE_UNRESTRICTED = False
    _API_VERSION = 'v1alpha'

    @staticmethod
    def Args(parser):
        UpdateAuthorizedOrgsDescsBase.ArgsVersioned(parser)