from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from apitools.base.py.extra_types import _JsonValueToPythonValue
from googlecloudsdk.api_lib.scc import securitycenter_client as sc_client
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
def _GetProjectId(asset):
    project_id = [x.value for x in asset.resourceProperties.additionalProperties if x.key == 'projectId']
    if not project_id:
        raise InvalidSCCInputError('No projectId exists for this asset.')
    return _JsonValueToPythonValue(project_id[0])