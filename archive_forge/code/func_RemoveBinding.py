from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict, Iterable, List, Optional, Set, TypedDict
import uuid
from apitools.base.py import encoding
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def RemoveBinding(to_resource: runapps_v1alpha1_messages.Resource, from_resource: runapps_v1alpha1_messages.Resource):
    """Remove a binding from a resource that's pointing to another resource.

  Args:
    to_resource: the resource this binding is pointing to.
    from_resource: the resource this binding is configured from.
  """
    from_resource.bindings = [x for x in from_resource.bindings if x.targetRef.id != to_resource.id]