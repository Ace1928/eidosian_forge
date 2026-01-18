from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import instance_template_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import partner_metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import resource_manager_tags_utils
from googlecloudsdk.command_lib.compute.instance_templates import flags as instance_templates_flags
from googlecloudsdk.command_lib.compute.instance_templates import mesh_util
from googlecloudsdk.command_lib.compute.instance_templates import service_proxy_aux_data
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.compute.resource_policies import flags as maintenance_flags
from googlecloudsdk.command_lib.compute.sole_tenancy import flags as sole_tenancy_flags
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
import six
def _AddSourceInstanceToTemplate(compute_api, args, instance_template, support_source_instance):
    """Set the source instance for the template."""
    if not support_source_instance or not args.source_instance:
        return
    source_instance_arg = instance_templates_flags.MakeSourceInstanceArg()
    source_instance_ref = source_instance_arg.ResolveAsResource(args, compute_api.resources)
    instance_template.sourceInstance = source_instance_ref.SelfLink()
    if args.configure_disk:
        messages = compute_api.client.messages
        instance_template.sourceInstanceParams = messages.SourceInstanceParams()
        for disk in args.configure_disk:
            disk_config = messages.DiskInstantiationConfig()
            disk_config.deviceName = disk.get('device-name')
            disk_config.autoDelete = disk.get('auto-delete')
            instantiate_from = disk.get('instantiate-from')
            if instantiate_from:
                disk_config.instantiateFrom = messages.DiskInstantiationConfig.InstantiateFromValueValuesEnum(disk.get('instantiate-from').upper().replace('-', '_'))
            disk_config.customImage = disk.get('custom-image')
            instance_template.sourceInstanceParams.diskConfigs.append(disk_config)
    instance_template.properties = None