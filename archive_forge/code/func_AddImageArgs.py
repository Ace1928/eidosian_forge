from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import functools
import ipaddress
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import disks_util
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as core_resources
import six
def AddImageArgs(parser, enable_snapshots=False, support_image_family_scope=False, enable_instant_snapshots=False):
    """Adds arguments related to images for instances and instance-templates."""

    def AddImageHelp():
        """Returns the detailed help for the `--image` flag."""
        return "\n          Specifies the boot image for the instances. For each\n          instance, a new boot disk will be created from the given\n          image. Each boot disk will have the same name as the\n          instance. To view a list of public images and projects, run\n          `$ gcloud compute images list`. It is best practice to use `--image`\n          when a specific version of an image is needed.\n\n          When using this option, ``--boot-disk-device-name'' and\n          ``--boot-disk-size'' can be used to override the boot disk's\n          device name and size, respectively.\n          "
    image_parent_group = parser.add_group()
    image_group = image_parent_group.add_mutually_exclusive_group()
    image_group.add_argument('--image', help=AddImageHelp, metavar='IMAGE')
    image_utils.AddImageProjectFlag(image_parent_group)
    image_group.add_argument('--image-family', help="      The image family for the operating system that the boot disk will\n      be initialized with. Compute Engine offers multiple Linux\n      distributions, some of which are available as both regular and\n      Shielded VM images.  When a family is specified instead of an image,\n      the latest non-deprecated image associated with that family is\n      used. It is best practice to use `--image-family` when the latest\n      version of an image is needed.\n\n      By default, ``{default_image_family}'' is assumed for this flag.\n      ".format(default_image_family=constants.DEFAULT_IMAGE_FAMILY))
    if enable_snapshots:
        image_group.add_argument('--source-snapshot', help='        The name of the source disk snapshot that the instance boot disk\n        will be created from. You can provide this as a full URL\n        to the snapshot or just the snapshot name. For example, the following\n        are valid values:\n\n          * https://compute.googleapis.com/compute/v1/projects/myproject/global/snapshots/snapshot\n          * snapshot\n        ')
    if enable_instant_snapshots:
        image_group.add_argument('--source-instant-snapshot', help='        The name of the source disk instant snapshot that the instance boot disk\n        will be created from. You can provide this as a full URL\n        to the instant snapshot. For example, the following is a valid value:\n\n          * https://compute.googleapis.com/compute/v1/projects/myproject/zones/my-zone/instantSnapshots/instant-snapshot\n        ')
    if support_image_family_scope:
        image_utils.AddImageFamilyScopeFlag(image_parent_group)