from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict, Iterable, List, Optional, Set, TypedDict
import uuid
from apitools.base.py import encoding
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def _GenerateIntegrationNameCandidate(self, integration_type: str) -> str:
    """Generates a suffix for a new integration.

    Args:
      integration_type: str, name of integration.

    Returns:
      str, a new name for the integration.
    """
    integration_suffix = str(uuid.uuid4())[:4]
    name = '{}-{}'.format(integration_type, integration_suffix)
    return name