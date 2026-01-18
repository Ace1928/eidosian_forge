from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict, Iterable, List, Optional, Set, TypedDict
import uuid
from apitools.base.py import encoding
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def GetBinding(self, res_id: runapps_v1alpha1_messages.ResourceID) -> List[BindingData]:
    """Returns list of bindings that are associated with the res_id.

    Args:
      res_id: the ID that represents the resource.

    Returns:
      list of binding data
    """
    return [b for b in self.bindings if b.from_id == res_id or b.to_id == res_id]