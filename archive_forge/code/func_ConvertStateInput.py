from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.scc import securitycenter_client
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util as scc_util
def ConvertStateInput(state, version):
    """Convert state input to messages.Finding.StateValueValuesEnum object."""
    messages = securitycenter_client.GetMessages(version)
    if state:
        state = state.upper()
    state_dict = {}
    if version == 'v1':
        unspecified_state = messages.Finding.StateValueValuesEnum.STATE_UNSPECIFIED
        state_dict['v1'] = {'ACTIVE': messages.Finding.StateValueValuesEnum.ACTIVE, 'INACTIVE': messages.Finding.StateValueValuesEnum.INACTIVE, 'STATE_UNSPECIFIED': unspecified_state}
    else:
        v2_unspecified_state = messages.GoogleCloudSecuritycenterV2Finding.StateValueValuesEnum.STATE_UNSPECIFIED
        state_dict['v2'] = {'ACTIVE': messages.GoogleCloudSecuritycenterV2Finding.StateValueValuesEnum.ACTIVE, 'INACTIVE': messages.GoogleCloudSecuritycenterV2Finding.StateValueValuesEnum.INACTIVE, 'STATE_UNSPECIFIED': v2_unspecified_state}
    return state_dict[version].get(state, state_dict[version]['STATE_UNSPECIFIED'])