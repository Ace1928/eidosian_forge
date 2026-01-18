from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.edge_cloud.container import resource_args as gdce_resource_args
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def GetInstanceResourcePresentationSpec():
    instance_data = yaml_data.ResourceYAMLData.FromPath('dataproc_gdc.service_instance')
    resource_spec = concepts.ResourceSpec.FromYaml(instance_data.GetData())
    return presentation_specs.ResourcePresentationSpec(name='instance', concept_spec=resource_spec, group_help='Name of the service instance to create.', required=True, prefixes=False)