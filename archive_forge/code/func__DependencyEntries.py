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
def _DependencyEntries(is_prebuilt_image=False, requirements_path=None, setup_path=None, extra_requirements=None, extra_packages=None, extra_dirs=None):
    """Returns the Dockerfile entries required to install dependencies.

  Args:
    is_prebuilt_image: (bool) Whether the base image is pre-built and provided
      by Vertex AI.
    requirements_path: (str) Path that points to a requirements.txt file
    setup_path: (str) Path that points to a setup.py
    extra_requirements: (List[str]) Required dependencies to be installed from
      remote resource archives.
    extra_packages: (List[str]) User custom dependency packages to install.
    extra_dirs: (List[str]) Directories other than the work_dir required.
  """
    ret = ''
    pip_version = 'pip3' if is_prebuilt_image else 'pip'
    if setup_path is not None:
        ret += textwrap.dedent('\n        {}\n        RUN {} install --no-cache-dir .\n        '.format(_GenerateCopyCommand(setup_path, './setup.py', comment='Found setup.py file, thus copy it to the docker container.'), pip_version))
    if requirements_path is not None:
        ret += textwrap.dedent('\n        {}\n        RUN {} install --no-cache-dir -r ./requirements.txt\n        '.format(_GenerateCopyCommand(requirements_path, './requirements.txt', comment='Found requirements.txt file, thus to the docker container.'), pip_version))
    if extra_packages is not None:
        for extra in extra_packages:
            package_name = os.path.basename(extra)
            ret += textwrap.dedent('\n        {}\n        RUN {} install --no-cache-dir {}\n        '.format(_GenerateCopyCommand(extra, package_name), pip_version, shlex_quote(package_name)))
    if extra_requirements is not None:
        for requirement in extra_requirements:
            ret += textwrap.dedent('\n        RUN {} install --no-cache-dir --upgrade {}\n        '.format(pip_version, shlex_quote(requirement)))
    if extra_dirs is not None:
        for directory in extra_dirs:
            ret += '\n{}\n'.format(_GenerateCopyCommand(directory, directory))
    return ret