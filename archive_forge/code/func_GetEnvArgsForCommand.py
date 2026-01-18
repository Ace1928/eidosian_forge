import copy
import json
import os
from googlecloudsdk.command_lib.anthos.common import messages
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions as c_except
from googlecloudsdk.core.credentials import store as c_store
def GetEnvArgsForCommand(extra_vars=None, exclude_vars=None):
    """Return an env dict to be passed on command invocation."""
    env = copy.deepcopy(os.environ)
    if extra_vars:
        env.update(extra_vars)
    if exclude_vars:
        for k in exclude_vars:
            env.pop(k)
    return env