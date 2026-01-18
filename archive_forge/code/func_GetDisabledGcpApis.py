from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Dict, List
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import services_util
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def GetDisabledGcpApis(self, project_id: str) -> List[str]:
    """Returns all GCP APIs needed for an integration.

    Args:
      project_id: The project's ID

    Returns:
      A list where each item is a GCP API that is not enabled.
    """
    required_apis = set(self.type_metadata.required_apis).union(types_utils.BASELINE_APIS)
    project_id = properties.VALUES.core.project.Get()
    apis_not_enabled = [api for api in sorted(required_apis) if not enable_api.IsServiceEnabled(project_id, api)]
    return apis_not_enabled