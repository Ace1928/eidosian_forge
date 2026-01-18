from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict, Iterable, List, Optional, Set, TypedDict
import uuid
from apitools.base.py import encoding
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def FindBindingsRecursive(resource: runapps_v1alpha1_messages.Resource, target_type: Optional[str]=None, target_name: Optional[str]=None) -> List[runapps_v1alpha1_messages.Binding]:
    """Find bindings from the given resource and its subresource.

  Args:
    resource: the resource to look for bindings from.
    target_type: the type of bindings to match. If empty, will match all types.
    target_name: the name of the bindings to match. If empty, will match all
      names.

  Returns:
    list of ResourceID of the bindings.
  """
    svcs = FindBindings(resource, target_type, target_name)
    if resource.subresources:
        for subresource in resource.subresources:
            svcs.extend(FindBindingsRecursive(subresource, target_type, target_name))
    return svcs