from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class AuthorizationToolkit(base.Group):
    """Manage Authorization Toolkit resources."""
    category = base.IDENTITY_AND_SECURITY_CATEGORY