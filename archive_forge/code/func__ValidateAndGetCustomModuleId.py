from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
from googlecloudsdk.command_lib.scc.hooks import CleanUpUserInput
def _ValidateAndGetCustomModuleId(args):
    """Validates customModuleId."""
    custom_module_id = args.custom_module
    if _ETD_CUSTOM_MODULE_ID_PATTERN.match(custom_module_id):
        return custom_module_id
    raise InvalidSCCInputError("Custom module ID does not match the pattern '%s'." % _ETD_CUSTOM_MODULE_ID_PATTERN.pattern)