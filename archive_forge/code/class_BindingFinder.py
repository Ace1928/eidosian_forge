from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict, Iterable, List, Optional, Set, TypedDict
import uuid
from apitools.base.py import encoding
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
class BindingFinder(object):
    """A map of bindings to help processing binding information.

  Attributes:
    bindings: the list of bindings.
  """

    def __init__(self, all_resources: List[runapps_v1alpha1_messages.Resource]):
        """Returns list of bindings between the given resources.

    Args:
      all_resources: the resources to look for bindings from.

    Returns:
      list of ResourceID of the bindings.
    """
        self.bindings = []
        for res in all_resources:
            bindings = FindBindingsRecursive(res)
            for binding in bindings:
                binding_data = BindingData(from_id=res.id, to_id=binding.targetRef.id, config=binding.config)
                self.bindings.append(binding_data)

    def GetAllBindings(self) -> List[runapps_v1alpha1_messages.ResourceID]:
        """Returns all the bindings.

    Returns:
      the list of bindings
    """
        return self.bindings

    def GetBinding(self, res_id: runapps_v1alpha1_messages.ResourceID) -> List[BindingData]:
        """Returns list of bindings that are associated with the res_id.

    Args:
      res_id: the ID that represents the resource.

    Returns:
      list of binding data
    """
        return [b for b in self.bindings if b.from_id == res_id or b.to_id == res_id]

    def GetIDsBindedTo(self, res_id: runapps_v1alpha1_messages.ResourceID) -> List[runapps_v1alpha1_messages.ResourceID]:
        """Returns list of resource IDs that are binded to the resource.

    Args:
      res_id: the ID that represents the resource.

    Returns:
      list of resource ID
    """
        return [bid.from_id for bid in self.GetBinding(res_id) if bid.to_id == res_id]

    def GetBindingIDs(self, res_id: runapps_v1alpha1_messages.ResourceID) -> List[runapps_v1alpha1_messages.ResourceID]:
        """Returns list of resource IDs that are binded to or from the resource.

    Args:
      res_id: the ID that represents the resource.

    Returns:
      list of resource ID
    """
        result = []
        for bid in self.GetBinding(res_id):
            if bid.from_id == res_id:
                result.append(bid.to_id)
            else:
                result.append(bid.from_id)
        return result