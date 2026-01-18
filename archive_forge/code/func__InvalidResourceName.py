from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
from googlecloudsdk.command_lib.scc.hooks import CleanUpUserInput
def _InvalidResourceName():
    """Returns an error indicating that a module lacks a valid resource name."""
    return InvalidSCCInputError('Custom module must match the full resource name, or `--organization=`, `--folder=`, or `--project=` must be provided.')