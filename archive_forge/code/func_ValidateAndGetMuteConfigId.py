from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util as scc_util
def ValidateAndGetMuteConfigId(args):
    """Validate muteConfigId."""
    mute_config_id = args.mute_config
    pattern = re.compile('^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$')
    if not pattern.match(mute_config_id):
        raise errors.InvalidSCCInputError("Mute config id does not match the pattern '^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$'.")
    else:
        return mute_config_id