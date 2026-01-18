from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _GetServerTlsPolicyResourcePresentationSpec(flag, noun, verb, required=False, plural=False, group=None, region_fallthrough=False):
    """Returns ResourcePresentationSpec for server TLS policy resource.

  Args:
    flag: str, the flag name.
    noun: str, the resource.
    verb: str, the verb to describe the resource, such as 'to update'.
    required: bool, if False, means that map ID is optional.
    plural: bool.
    group: args group.
    region_fallthrough: bool, True if the command has a region flag that should
      be used as a fallthrough for the server TLS policy location.

  Returns:
    presentation_specs.ResourcePresentationSpec.
  """
    flag_overrides = {'location': ''}
    return presentation_specs.ResourcePresentationSpec(flag, _GetServerTlsPolicyResourceSpec(region_fallthrough), '{} {}.'.format(noun, verb), required=required, plural=plural, group=group, flag_name_overrides=flag_overrides)