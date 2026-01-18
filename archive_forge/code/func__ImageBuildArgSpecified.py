from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.ai import util as api_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import validation
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.command_lib.ai.docker import utils as docker_utils
from googlecloudsdk.core.util import files
def _ImageBuildArgSpecified(args):
    """Returns names of all the flags specified only for image building."""
    image_build_args = []
    if args.script:
        image_build_args.append('script')
    if args.python_module:
        image_build_args.append('python-module')
    if args.requirements:
        image_build_args.append('requirements')
    if args.extra_packages:
        image_build_args.append('extra-packages')
    if args.extra_dirs:
        image_build_args.append('extra-dirs')
    if args.output_image_uri:
        image_build_args.append('output-image-uri')
    return image_build_args