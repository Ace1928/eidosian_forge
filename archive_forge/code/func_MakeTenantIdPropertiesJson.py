from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import re
from typing import Any, List, NamedTuple, Optional
from utils import bq_error
from utils import bq_id_utils
def MakeTenantIdPropertiesJson(tenant_id: str) -> str:
    """Returns properties for a connection with tenant id.

  Args:
    tenant_id: tenant id.

  Returns:
    JSON string with properties to create a connection with customer's tenant
    id.
  """
    return '{"customerTenantId": "%s"}' % tenant_id