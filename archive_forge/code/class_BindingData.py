from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict, Iterable, List, Optional, Set, TypedDict
import uuid
from apitools.base.py import encoding
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
class BindingData(object):
    """Binding data that represent a binding.

  Attributes:
    from_id: the resource id the binding is configured from
    to_id: the resource id the binding is pointing to
    config: the binding config if available
  """

    def __init__(self, from_id: runapps_v1alpha1_messages.ResourceID, to_id: runapps_v1alpha1_messages.ResourceID, config: Optional[runapps_v1alpha1_messages.Binding.ConfigValue]=None):
        self.from_id = from_id
        self.to_id = to_id
        self.config = config