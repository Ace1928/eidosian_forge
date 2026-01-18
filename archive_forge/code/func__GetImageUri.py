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
def _GetImageUri(self, args, client, holder, instance_template_ref):
    if args.IsSpecified('image') or args.IsSpecified('image_family') or args.IsSpecified('image_project'):
        image_expander = image_utils.ImageExpander(client, holder.resources)
        image_uri, _ = image_expander.ExpandImageFlag(user_project=instance_template_ref.project, image=args.image, image_family=args.image_family, image_project=args.image_project)
        if holder.resources.Parse(image_uri).project != 'cos-cloud':
            log.warning('This container deployment mechanism requires a Container-Optimized OS image in order to work. Select an image from a cos-cloud project (cos-stable, cos-beta, cos-dev image families).')
    else:
        image_uri = containers_utils.ExpandKonletCosImageFlag(client)
    return image_uri