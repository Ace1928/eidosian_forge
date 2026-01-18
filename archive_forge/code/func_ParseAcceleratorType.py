from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.compute.sole_tenancy.node_templates import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
import six
def ParseAcceleratorType(accelerator_type_name, resource_parser, project, region):
    collection = 'compute.regionAcceleratorTypes'
    params = {'project': project, 'region': region}
    accelerator_type = resource_parser.Parse(accelerator_type_name, collection=collection, params=params).SelfLink()
    return accelerator_type