from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.GA)
class VirtualEnv(base.Group):
    """Manage Cloud SDK virtual env setup."""