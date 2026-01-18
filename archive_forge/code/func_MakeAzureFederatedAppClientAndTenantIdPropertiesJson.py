from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import re
from typing import Any, List, NamedTuple, Optional
from utils import bq_error
from utils import bq_id_utils
def MakeAzureFederatedAppClientAndTenantIdPropertiesJson(tenant_id: str, federated_app_client_id: str) -> str:
    """Returns properties for a connection with tenant and federated app ids.

  Args:
    tenant_id: tenant id
    federated_app_client_id: federated application (client) id.

  Returns:
    JSON string with properties to create a connection with customer's tenant
    and federated application (client) ids.
  """
    return '{"customerTenantId": "%s", "federatedApplicationClientId" : "%s"}' % (tenant_id, federated_app_client_id)