from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddPostgresqlConnectionProfileResourceArg(parser, verb, positional=True):
    """Add a resource argument for a database migration postgresql cp.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
    positional: bool, if True, means that the resource is a positional rather
      than a flag.
  """
    if positional:
        name = 'connection_profile'
    else:
        name = '--connection-profile'
    connectivity_parser = parser.add_group(mutex=True)
    connectivity_parser.add_argument('--static-ip-connectivity', action='store_true', help='use static ip connectivity')
    resource_specs = [presentation_specs.ResourcePresentationSpec(name, GetConnectionProfileResourceSpec(), 'The connection profile {}.'.format(verb), required=True), presentation_specs.ResourcePresentationSpec('--psc-service-attachment', GetServiceAttachmentResourceSpec(), 'Resource ID of the service attachment.', flag_name_overrides={'region': ''}, group=connectivity_parser)]
    concept_parsers.ConceptParser(resource_specs, command_level_fallthroughs={'--psc-service-attachment.region': ['--region']}).AddToParser(parser)