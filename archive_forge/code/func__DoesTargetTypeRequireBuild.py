import gyp.common
import json
import os
import posixpath
def _DoesTargetTypeRequireBuild(target_dict):
    """Returns true if the target type is such that it needs to be built."""
    return bool(target_dict['type'] != 'none' or target_dict.get('actions') or target_dict.get('rules'))