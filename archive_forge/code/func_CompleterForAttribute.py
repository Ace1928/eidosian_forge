from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.util import resource as resource_lib  # pylint: disable=unused-import
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.concepts import resource_parameter_info
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def CompleterForAttribute(resource_spec, attribute_name):
    """Gets a resource argument completer for a specific attribute."""

    class Completer(ResourceArgumentCompleter):
        """A specific completer for this attribute and resource."""

        def __init__(self, resource_spec=resource_spec, attribute_name=attribute_name, **kwargs):
            completer_info = CompleterInfo.FromResource(resource_spec, attribute_name)
            super(Completer, self).__init__(resource_spec, completer_info.collection_info, completer_info.method, static_params=completer_info.static_params, id_field=completer_info.id_field, param=completer_info.param_name, **kwargs)

        @classmethod
        def validate(cls):
            """Checks whether the completer is valid (has a list method)."""
            return bool(CompleterInfo.FromResource(resource_spec, attribute_name).GetMethod())
    if not Completer.validate():
        return None
    return Completer