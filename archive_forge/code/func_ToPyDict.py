from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from typing import Generator
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as messages
@staticmethod
def ToPyDict(proto_map_value):
    """Helper to convert proto map Values to normal dictionaries.

    encoding.MessageToPyValue recursively converts values to dicts, while this
    method leaves the map values as proto objects.

    Args:
      proto_map_value: The map field "Value". For example, the `Feature.labels`
        value (of type `Features.LabelsValue`). Can be None.

    Returns:
      An OrderedDict of the map's keys/values, in the original order.
    """
    if proto_map_value is None or proto_map_value.additionalProperties is None:
        return {}
    return collections.OrderedDict(((p.key, p.value) for p in proto_map_value.additionalProperties))