from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
from googlecloudsdk.command_lib.scc.hooks import CleanUpUserInput
def _ValidateAndGetCustomModuleFullResourceName(args):
    """Validates a custom module's full resource name."""
    custom_module = args.custom_module
    if _ETD_CUSTOM_MODULE_PATTERN.match(custom_module):
        return custom_module
    raise _InvalidResourceName()