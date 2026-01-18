from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class AuditScopes(base.Group):
    """Command group for Audit Manager Audit Scopes."""
    category = base.SECURITY_CATEGORY