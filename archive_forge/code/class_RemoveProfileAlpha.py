from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.oslogin import client
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class RemoveProfileAlpha(RemoveProfile):
    """Remove the posix account information for the current user."""

    @staticmethod
    def Args(parser):
        os_arg = base.ChoiceArgument('--operating-system', choices=('linux', 'windows'), required=False, default='linux', help_str='Specifies the profile type to remove.')
        os_arg.AddToParser(parser)