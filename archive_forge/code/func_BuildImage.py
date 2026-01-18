from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import posixpath
import re
import textwrap
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.command_lib.ai.docker import utils
from googlecloudsdk.core import log
from six.moves import shlex_quote
def BuildImage(base_image, host_workdir, main_script, output_image_name, python_module=None, requirements=None, extra_packages=None, container_workdir=None, container_home=None, no_cache=True, **kwargs):
    """Builds a Docker image.

  Generates a Dockerfile and passes it to `docker build` via stdin.
  All output from the `docker build` process prints to stdout.

  Args:
    base_image: (str) ID or name of the base image to initialize the build
      stage.
    host_workdir: (str) A path indicating where all the required sources
      locates.
    main_script: (str) A string that identifies the executable script under the
      working directory.
    output_image_name: (str) Name of the built image.
    python_module: (str) Represents the executable main_script in form of a
      python module, if applicable.
    requirements: (List[str]) Required dependencies to install from PyPI.
    extra_packages: (List[str]) User custom dependency packages to install.
    container_workdir: (str) Working directory in the container.
    container_home: (str) the $HOME directory in the container.
    no_cache: (bool) Do not use cache when building the image.
    **kwargs: Other arguments to pass to underlying method that generates the
      Dockerfile.

  Returns:
    A Image class that contains info of the built image.

  Raises:
    DockerError: An error occurred when executing `docker build`
  """
    tag_options = ['-t', output_image_name]
    cache_args = ['--no-cache'] if no_cache else []
    command = ['docker', 'build'] + cache_args + tag_options + ['--rm', '-f-', host_workdir]
    has_setup_py = os.path.isfile(os.path.join(host_workdir, _DEFAULT_SETUP_PATH))
    setup_path = _DEFAULT_SETUP_PATH if has_setup_py else None
    has_requirements_txt = os.path.isfile(os.path.join(host_workdir, _DEFAULT_REQUIREMENTS_PATH))
    requirements_path = _DEFAULT_REQUIREMENTS_PATH if has_requirements_txt else None
    home_dir = container_home or _DEFAULT_HOME
    work_dir = container_workdir or _DEFAULT_WORKDIR
    main_package = utils.Package(script=main_script.replace(os.sep, posixpath.sep), package_path=host_workdir.replace(os.sep, posixpath.sep), python_module=python_module)
    dockerfile = _MakeDockerfile(base_image, main_package=main_package, container_home=home_dir, container_workdir=work_dir, requirements_path=requirements_path, setup_path=setup_path, extra_requirements=requirements, extra_packages=extra_packages, **kwargs)
    joined_command = ' '.join(command)
    log.info('Running command: {}'.format(joined_command))
    return_code = local_util.ExecuteCommand(command, input_str=dockerfile)
    if return_code == 0:
        return utils.Image(output_image_name, home_dir, work_dir)
    else:
        error_msg = textwrap.dedent('\n        Docker failed with error code {code}.\n        Command: {cmd}\n        '.format(code=return_code, cmd=joined_command))
        raise errors.DockerError(error_msg, command, return_code)