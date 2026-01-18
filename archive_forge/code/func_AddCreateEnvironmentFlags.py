from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.compute.networks import flags as compute_network_flags
from googlecloudsdk.command_lib.compute.networks.subnets import flags as compute_subnet_flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.notebooks import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddCreateEnvironmentFlags(api_version, parser):
    """Construct groups and arguments specific to the environment create."""
    source_group = parser.add_group(mutex=True, required=True)
    vm_source_group = source_group.add_group()
    vm_mutex_group = vm_source_group.add_group(mutex=True, required=True)
    container_group = source_group.add_group()
    GetEnvironmentResourceArg(api_version, 'User-defined unique name of this environment. The environment name must be 1 to 63 characters long and contain only lowercase letters, numeric characters, and dashes. The first character must be a lowercaseletter and the last character cannot be a dash.').AddToParser(parser)
    parser.add_argument('--description', help='A brief description of this environment.')
    parser.add_argument('--display-name', help='Name to display on the UI.')
    parser.add_argument('--post-startup-script', help='Path to a Bash script that automatically runs after a notebook instance fully boots up. The path must be a URL or Cloud Storage path(gs://`path-to-file/`file-name`).')
    base.ASYNC_FLAG.AddToParser(parser)
    vm_source_group.add_argument('--vm-image-project', help='The ID of the Google Cloud project that this VM image belongs to.Format: projects/`{project_id}`.', default='deeplearning-platform-release')
    vm_mutex_group.add_argument('--vm-image-family', help='Use this VM image family to find the image; the newest image in this family will be used.', default='common-cpu')
    vm_mutex_group.add_argument('--vm-image-name', help='Use this VM image name to find the image.')
    container_group.add_argument('--container-repository', help='The path to the container image repository. For example: gcr.io/`{project_id}`/`{image_name}`.', required=True)
    container_group.add_argument('--container-tag', help='The tag of the container image. If not specified, this defaults to the latest tag.')