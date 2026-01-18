from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict, Iterable, List, Optional, Set, TypedDict
import uuid
from apitools.base.py import encoding
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def GetAllBindings(self) -> List[runapps_v1alpha1_messages.ResourceID]:
    """Returns all the bindings.

    Returns:
      the list of bindings
    """
    return self.bindings