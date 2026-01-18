import gyp.common
import json
import os
import posixpath
def _GetTargetsDependingOnMatchingTargets(possible_targets):
    """Returns the list of Targets in |possible_targets| that depend (either
  directly on indirectly) on at least one of the targets containing the files
  supplied as input to analyzer.
  possible_targets: targets to search from."""
    found = []
    print('Targets that matched by dependency:')
    for target in possible_targets:
        if _DoesTargetDependOnMatchingTargets(target):
            found.append(target)
    return found