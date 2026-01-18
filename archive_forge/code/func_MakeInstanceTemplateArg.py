from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.instance_templates import service_proxy_aux_data
def MakeInstanceTemplateArg(plural=False, include_regional=False):
    return flags.ResourceArgument(resource_name='instance template', completer=completers.InstanceTemplatesCompleter, plural=plural, global_collection='compute.instanceTemplates', regional_collection='compute.regionInstanceTemplates' if include_regional else None)