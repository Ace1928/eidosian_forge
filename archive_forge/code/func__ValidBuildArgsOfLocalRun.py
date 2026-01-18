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
def _ValidBuildArgsOfLocalRun(args):
    """Validates the arguments related to image building and normalize them."""
    build_args_specified = _ImageBuildArgSpecified(args)
    if not build_args_specified:
        return
    if not args.script and (not args.python_module):
        raise exceptions.MinimumArgumentException(['--script', '--python-module'], 'They are required to build a training container image. Otherwise, please remove flags [{}] to directly run the `executor-image-uri`.'.format(', '.join(sorted(build_args_specified))))
    if args.script:
        arg_name = '--script'
    else:
        args.script = local_util.ModuleToPath(args.python_module)
        arg_name = '--python-module'
    script_path = os.path.normpath(os.path.join(args.local_package_path, args.script))
    if not os.path.exists(script_path) or not os.path.isfile(script_path):
        raise exceptions.InvalidArgumentException(arg_name, "File '{}' is not found under the package: '{}'.".format(args.script, args.local_package_path))
    for package in args.extra_packages or []:
        package_path = os.path.normpath(os.path.join(args.local_package_path, package))
        if not os.path.exists(package_path) or not os.path.isfile(package_path):
            raise exceptions.InvalidArgumentException('--extra-packages', "Package file '{}' is not found under the package: '{}'.".format(package, args.local_package_path))
    for directory in args.extra_dirs or []:
        dir_path = os.path.normpath(os.path.join(args.local_package_path, directory))
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            raise exceptions.InvalidArgumentException('--extra-dirs', "Directory '{}' is not found under the package: '{}'.".format(directory, args.local_package_path))
    if args.output_image_uri:
        output_image = args.output_image_uri
        try:
            docker_utils.ValidateRepositoryAndTag(output_image)
        except ValueError as e:
            raise exceptions.InvalidArgumentException('--output-image-uri', "'{}' is not a valid container image uri: {}".format(output_image, e))
    else:
        args.output_image_uri = docker_utils.GenerateImageName(base_name=args.script)