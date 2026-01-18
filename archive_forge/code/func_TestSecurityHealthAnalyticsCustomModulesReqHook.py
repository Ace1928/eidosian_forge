from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
from googlecloudsdk.command_lib.scc.hooks import CleanUpUserInput
def TestSecurityHealthAnalyticsCustomModulesReqHook(ref, args, req):
    """Test a Security Health Analytics custom module."""
    del ref
    parent = _ValidateAndGetParent(args)
    custom_module_name = ''
    test_req = req.testSecurityHealthAnalyticsCustomModuleRequest
    if parent is not None:
        custom_module_id = _ValidateAndGetCustomModuleId(args)
        custom_module_name = parent + '/customModules/' + custom_module_id
    else:
        custom_module_name = _ValidateAndGetCustomModuleFullResourceName(args)
    req.name = custom_module_name
    if test_req.securityHealthAnalyticsCustomModule is not None:
        test_req.securityHealthAnalyticsCustomModule.name = custom_module_name
    return req