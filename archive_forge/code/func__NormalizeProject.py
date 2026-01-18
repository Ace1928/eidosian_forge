from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
from googlecloudsdk.command_lib.scc.hooks import CleanUpUserInput
def _NormalizeProject(project):
    """Validates and normalizes a project name."""
    if '/' in project:
        if _PROJECT_NAME_PATTERN.fullmatch(project):
            return project + _PARENT_SUFFIX
        raise _InvalidFullResourcePathForPattern(_PROJECT_NAME_PATTERN)
    return 'projects/' + project + _PARENT_SUFFIX