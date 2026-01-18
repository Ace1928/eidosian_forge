from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import deps_map_util
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class ResourceParameterAttributeConfig(object):
    """Configuration used to create attributes from resource parameters."""

    @classmethod
    def FromData(cls, data):
        """Constructs an attribute config from data defined in the yaml file.

    Args:
      data: {}, the dict of data from the YAML file for this single attribute.

    Returns:
      ResourceParameterAttributeConfig
    """
        attribute_name = data['attribute_name']
        parameter_name = data['parameter_name']
        help_text = data['help']
        completer = util.Hook.FromData(data, 'completer')
        completion_id_field = data.get('completion_id_field', None)
        completion_request_params_list = data.get('completion_request_params', [])
        completion_request_params = {param.get('fieldName'): param.get('value') for param in completion_request_params_list}
        fallthroughs = []
        prop = properties.FromString(data.get('property', ''))
        if prop:
            fallthroughs.append(deps_lib.PropertyFallthrough(prop))
        default_config = DEFAULT_RESOURCE_ATTRIBUTE_CONFIGS.get(attribute_name)
        if default_config:
            fallthroughs += [f for f in default_config.fallthroughs if f not in fallthroughs]
        fallthrough_data = data.get('fallthroughs', [])
        fallthroughs_from_hook = []
        for f in fallthrough_data:
            if 'value' in f:
                fallthroughs_from_hook.append(deps_lib.ValueFallthrough(f['value'], f['hint'] if 'hint' in f else None))
            elif 'hook' in f:
                fallthroughs_from_hook.append(deps_lib.Fallthrough(util.Hook.FromPath(f['hook']), hint=f['hint']))
        fallthroughs += fallthroughs_from_hook
        return cls(name=attribute_name, help_text=help_text, fallthroughs=fallthroughs, completer=completer, completion_id_field=completion_id_field, completion_request_params=completion_request_params, parameter_name=parameter_name)

    def __init__(self, name=None, help_text=None, fallthroughs=None, completer=None, completion_request_params=None, completion_id_field=None, value_type=None, parameter_name=None):
        """Create a resource attribute.

    Args:
      name: str, the name of the attribute. This controls the naming of flags
        based on the attribute.
      help_text: str, generic help text for any flag based on the attribute. One
        special expansion is available to convert "{resource}" to the name of
        the resource.
      fallthroughs: [deps_lib.Fallthrough], A list of fallthroughs to use to
        resolve the attribute if it is not provided on the command line.
      completer: core.cache.completion_cache.Completer, the completer
        associated with the attribute.
      completion_request_params: {str: value}, a dict of field names to static
        values to fill in for the completion request.
      completion_id_field: str, the ID field of the return value in the
        response for completion commands.
      value_type: the type to be accepted by the attribute arg. Defaults to str.
      parameter_name: the API parameter name that this attribute maps to.
    """
        self.attribute_name = name
        self.help_text = help_text
        self.fallthroughs = fallthroughs or []
        if completer and (completion_request_params or completion_id_field):
            raise ValueError('Custom completer and auto-completer should not be specified at the same time')
        self.completer = completer
        self.completion_request_params = completion_request_params
        self.completion_id_field = completion_id_field
        self.value_type = value_type or str
        self.parameter_name = parameter_name