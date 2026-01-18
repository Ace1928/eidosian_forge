from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
import six
def GetResourcePresentationSpec(name, verb, resource_data, attribute_overrides=None, help_text='The {name} {verb}.', required=False, prefixes=True, positional=False):
    """Build ResourcePresentationSpec for a Resource.

  Args:
    name: string, name of resource anchor argument.
    verb: string, the verb to describe the resource, such as 'to create'.
    resource_data: dict, the parsed data from a resources.yaml file under
        command_lib/.
    attribute_overrides: dict{string:string}, map of resource attribute names to
      override in the generated resrouce spec.
    help_text: string, the help text for the entire resource arg group. Should
      have 2 format format specifiers (`{name}`, `{verb}`) to insert the
      name and verb repectively.
    required: bool, whether or not this resource arg is required.
    prefixes: bool, if True the resource name will be used as a prefix for
      the flags in the resource group.
    positional: bool, if True, means that the resource arg is a positional
      rather than a flag.
  Returns:
    ResourcePresentationSpec, presentation spec for resource.
  """
    arg_name = name if positional else '--' + name
    arg_help = help_text.format(verb=verb, name=name)
    if attribute_overrides:
        for attribute_name, value in six.iteritems(attribute_overrides):
            for attr in resource_data['attributes']:
                if attr['attribute_name'] == attribute_name:
                    attr['attribute_name'] = value
    return presentation_specs.ResourcePresentationSpec(arg_name, concepts.ResourceSpec.FromYaml(resource_data), arg_help, required=required, prefixes=prefixes)