from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.compute.resource_policies import flags as maintenance_flags
from googlecloudsdk.command_lib.util.args import labels_util
def MakeBulkSourceInstanceTemplateArg():
    return compute_flags.ResourceArgument(name='--source-instance-template', resource_name='instance template', completer=compute_completers.InstanceTemplatesCompleter, required=False, global_collection='compute.instanceTemplates', short_help='The name of the instance template that the instance will be created from. Users can override fields by specifying other flags.')