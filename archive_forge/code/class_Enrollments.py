from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Enrollments(base.Group):
    """Command group for Audit Manager Enrollments."""
    category = base.SECURITY_CATEGORY