from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AppendParentArg():
    """Add Parent as a positional resource."""
    parent_spec_data = {'name': 'parent', 'collection': 'securitycenter.organizations', 'attributes': [{'parameter_name': 'organizationsId', 'attribute_name': 'parent', 'help': '(Optional) Provide the full resource name,\n          [RESOURCE_TYPE/RESOURCE_ID], of the parent organization, folder, or\n          project resource. For example, `organizations/123` or `parent/456`.\n          If the parent is an organization, you can specify just the\n          organization ID. For example, `123`.', 'fallthroughs': [{'hook': 'googlecloudsdk.command_lib.scc.flags:GetDefaultParent', 'hint': 'Set the parent property in configuration using `gcloud\n              config set scc/parent` if it is not specified in command line'}]}], 'disable_auto_completers': 'false'}
    arg_specs = [resource_args.GetResourcePresentationSpec(verb='to be used for the `gcloud scc` command', name='parent', help_text='{name} organization, folder, or project in the Google Cloud resource hierarchy {verb}. Specify the argument as either [RESOURCE_TYPE/RESOURCE_ID] or [RESOURCE_ID], as shown in the preceding examples.', required=True, prefixes=False, positional=True, resource_data=parent_spec_data)]
    return [concept_parsers.ConceptParser(arg_specs, [])]