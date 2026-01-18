from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def AddBuildConfigArgs(flag_config, add_docker_args=True, require_docker_image=False):
    """Adds additional argparse flags to a group for build configuration options.

  Args:
    flag_config: argparse argument group. Additional flags will be added to this
      group to cover common build configuration settings.
    add_docker_args: If true, docker args are added automatically.
    require_docker_image: If true, --dockerfile-image must be provided.
  Returns:
    build_config: a build config.
  """
    AddSubstitutions(flag_config)
    build_config = flag_config.add_mutually_exclusive_group(required=True)
    build_config.add_argument('--build-config', metavar='PATH', help='Path to a YAML or JSON file containing the build configuration in the repository.\n\nFor more details, see: https://cloud.google.com/cloud-build/docs/build-config\n')
    build_config.add_argument('--inline-config', metavar='PATH', help='      Local path to a YAML or JSON file containing a build configuration.\n    ')
    if add_docker_args:
        AddBuildDockerArgs(build_config, require_docker_image=require_docker_image)
    return build_config