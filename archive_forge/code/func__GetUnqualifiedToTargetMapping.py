import gyp.common
import json
import os
import posixpath
def _GetUnqualifiedToTargetMapping(all_targets, to_find):
    """Returns a tuple of the following:
  . mapping (dictionary) from unqualified name to Target for all the
    Targets in |to_find|.
  . any target names not found. If this is empty all targets were found."""
    result = {}
    if not to_find:
        return ({}, [])
    to_find = set(to_find)
    for target_name in all_targets.keys():
        extracted = gyp.common.ParseQualifiedTarget(target_name)
        if len(extracted) > 1 and extracted[1] in to_find:
            to_find.remove(extracted[1])
            result[extracted[1]] = all_targets[target_name]
            if not to_find:
                return (result, [])
    return (result, [x for x in to_find])