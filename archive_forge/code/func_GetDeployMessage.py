from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict, Iterable, List, Optional, Set, TypedDict
import uuid
from apitools.base.py import encoding
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def GetDeployMessage(self, create: bool=False) -> str:
    """Message that is shown to the user upon starting the deployment.

    Each TypeKit should override this method to at least tell the user how
    long the deployment is expected to take.

    Args:
      create: denotes if the command was a create deployment.

    Returns:
      The message displayed to the user.
    """
    del create
    if self._type_metadata.eta_in_min:
        return 'This might take up to {} minutes.'.format(self._type_metadata.eta_in_min)
    return ''