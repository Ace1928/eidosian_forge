from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core.util import scaled_integer
import six
def MakeNodeTemplateArg():
    return compute_flags.ResourceArgument(resource_name='node templates', regional_collection='compute.nodeTemplates', region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION)