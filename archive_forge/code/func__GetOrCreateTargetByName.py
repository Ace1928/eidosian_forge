import gyp.common
import json
import os
import posixpath
def _GetOrCreateTargetByName(targets, target_name):
    """Creates or returns the Target at targets[target_name]. If there is no
  Target for |target_name| one is created. Returns a tuple of whether a new
  Target was created and the Target."""
    if target_name in targets:
        return (False, targets[target_name])
    target = Target(target_name)
    targets[target_name] = target
    return (True, target)