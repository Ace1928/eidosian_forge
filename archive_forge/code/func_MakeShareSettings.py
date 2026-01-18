from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.reservations import util as reservation_util
from googlecloudsdk.core.util import times
def MakeShareSettings(messages, args, setting_configs):
    """Constructs the share settings message object."""
    if setting_configs:
        if setting_configs == 'local':
            if args.IsSpecified('share_with'):
                raise exceptions.InvalidArgumentException('--share_with', 'The scope this reservation is to be shared with must not be specified with share setting local.')
            return messages.ShareSettings(shareType=messages.ShareSettings.ShareTypeValueValuesEnum.LOCAL)
        if setting_configs == 'projects':
            if not args.IsSpecified('share_with'):
                raise exceptions.InvalidArgumentException('--share_with', 'The projects this reservation is to be shared with must be specified.')
            return messages.ShareSettings(shareType=messages.ShareSettings.ShareTypeValueValuesEnum.SPECIFIC_PROJECTS, projects=getattr(args, 'share_with', None))
    else:
        return None