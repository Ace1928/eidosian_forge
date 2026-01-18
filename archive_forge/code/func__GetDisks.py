from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import instance_template_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.instances.create import utils as create_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.instance_templates import flags as instance_templates_flags
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
def _GetDisks(self, args, client, holder, instance_template_ref, image_uri, match_container_mount_disks=False):
    boot_disk_size_gb = self._GetBootDiskSize(args)
    create_boot_disk = not instance_utils.UseExistingBootDisk((args.disk or []) + (args.create_disk or []))
    return instance_template_utils.CreateDiskMessages(args, client, holder.resources, instance_template_ref.project, image_uri, boot_disk_size_gb, create_boot_disk=create_boot_disk, match_container_mount_disks=match_container_mount_disks)