from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict, Iterable, List, Optional, Set, TypedDict
import uuid
from apitools.base.py import encoding
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def BindServiceToIntegration(self, integration: runapps_v1alpha1_messages.Resource, workload: runapps_v1alpha1_messages.Resource, parameters: Optional[Dict[str, str]]=None):
    """Bind a workload to an integration.

    Args:
      integration: the resource of the inetgration.
      workload: the resource the workload.
      parameters: the binding config from parameter.
    """
    if self._type_metadata.service_type == types_utils.ServiceType.INGRESS:
        self._SetBinding(workload, integration, parameters)
    else:
        self._SetBinding(integration, workload, parameters)