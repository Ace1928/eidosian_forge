from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict, Iterable, List, Optional, Set, TypedDict
import uuid
from apitools.base.py import encoding
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def _SetBinding(self, to_resource: runapps_v1alpha1_messages.Resource, from_resource: runapps_v1alpha1_messages.Resource, parameters: Optional[Dict[str, str]]=None):
    """Add a binding from a resource to another resource.

    Args:
      to_resource: the resource this binding to be pointing to.
      from_resource: the resource this binding to be configured from.
      parameters: the binding config from parameter
    """
    from_ids = [x.targetRef.id for x in from_resource.bindings]
    if to_resource.id not in from_ids:
        from_resource.bindings.append(runapps_v1alpha1_messages.Binding(targetRef=runapps_v1alpha1_messages.ResourceRef(id=to_resource.id)))
    if parameters:
        for binding in from_resource.bindings:
            if binding.targetRef.id == to_resource.id:
                binding_config = encoding.MessageToDict(binding.config) if binding.config else {}
                for key in parameters:
                    binding_config[key] = parameters[key]
                binding.config = encoding.DictToMessage(binding_config, runapps_v1alpha1_messages.Binding.ConfigValue)