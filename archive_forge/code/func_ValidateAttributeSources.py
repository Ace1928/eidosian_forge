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
def ValidateAttributeSources(self, aggregations):
    """Validates that parent attributes values exitst before making request."""
    parameters_needing_resolution = set([p.name for p in self.parameters[:-1]])
    resolved_parameters = set([a.name for a in aggregations])
    for attribute in self.resource_spec.attributes:
        if CompleterForAttribute(self.resource_spec, attribute.name):
            resolved_parameters.add(self.resource_spec.attribute_to_params_map[attribute.name])
    return parameters_needing_resolution.issubset(resolved_parameters)