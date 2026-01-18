from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import textwrap
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import disks_util
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute import zone_utils
from googlecloudsdk.api_lib.compute.regions import utils as region_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.disks import create
from googlecloudsdk.command_lib.compute.disks import flags as disks_flags
from googlecloudsdk.command_lib.compute.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.compute.resource_policies import flags as resource_flags
from googlecloudsdk.command_lib.compute.resource_policies import util as resource_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
import six
def GetProjectToSourceImageDict(self, args, disk_refs, compute_holder, from_image):
    project_to_source_image = {}
    image_expander = image_utils.ImageExpander(compute_holder.client, compute_holder.resources)
    for disk_ref in disk_refs:
        if from_image:
            if disk_ref.project not in project_to_source_image:
                source_image_uri, _ = image_expander.ExpandImageFlag(user_project=disk_ref.project, image=args.image, image_family=args.image_family, image_project=args.image_project, return_image_resource=False, image_family_scope=args.image_family_scope, support_image_family_scope=True)
                project_to_source_image[disk_ref.project] = argparse.Namespace()
                project_to_source_image[disk_ref.project].uri = source_image_uri
        else:
            project_to_source_image[disk_ref.project] = argparse.Namespace()
            project_to_source_image[disk_ref.project].uri = None
    return project_to_source_image