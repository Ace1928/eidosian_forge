from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def CreateDevicePresentationSpec(verb, help_text='The device {}.', name='device', required=False, prefixes=True, positional=False):
    """Build ResourcePresentationSpec for generic device Resource.

  NOTE: Should be used when there are multiple resources args in the command.

  Args:
    verb: string, the verb to describe the resource, such as 'to bind'.
    help_text: string, the help text for the entire resource arg group. Should
      have a format specifier (`{}`) to insert verb.
    name: string, name of resource anchor argument.
    required: bool, whether or not this resource arg is required.
    prefixes: bool, if True the resource name will be used as a prefix for
      the flags in the resource group.
    positional: bool, if True, means that the device ID is a positional rather
      than a flag.
  Returns:
    ResourcePresentationSpec, presentation spec for device.
  """
    arg_name = name if positional else '--' + name
    arg_help = help_text.format(verb)
    return presentation_specs.ResourcePresentationSpec(arg_name, GetDeviceResourceSpec(name), arg_help, required=required, prefixes=prefixes)