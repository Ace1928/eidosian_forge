from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util as scc_util
def GenerateMuteConfigName(args, req, version):
    """Generates the name of the mute config."""
    parent = ValidateAndGetParent(args)
    if parent is not None:
        if version == 'v2':
            parent = ValidateAndGetRegionalizedParent(args, parent)
        mute_config_id = ValidateAndGetMuteConfigId(args)
        req.name = f'{parent}/muteConfigs/{mute_config_id}'
    else:
        args.location = scc_util.ValidateAndGetLocation(args, version)
        mute_config = ValidateAndGetMuteConfigFullResourceName(args, version)
        req.name = mute_config
    return req