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
def _ValidateWorkerPoolSoftwareWithoutLocalPackages(spec):
    """Validate the software in a single `--worker-pool-spec` when `local-package-path` is not specified."""
    assert 'local-package-path' not in spec
    has_executor_image = 'executor-image-uri' in spec
    has_container_image = 'container-image-uri' in spec
    has_python_module = 'python-module' in spec
    if has_executor_image + has_container_image != 1:
        raise exceptions.InvalidArgumentException('--worker-pool-spec', 'Exactly one of keys [executor-image-uri, container-image-uri] is required.')
    if has_container_image and has_python_module:
        raise exceptions.InvalidArgumentException('--worker-pool-spec', 'Key [python-module] is not allowed together with key [container-image-uri].')
    if has_executor_image and (not has_python_module):
        raise exceptions.InvalidArgumentException('--worker-pool-spec', 'Key [python-module] is required.')
    local_package_only_keys = {'script', 'requirements', 'extra-packages', 'extra-dirs'}
    unexpected_keys = local_package_only_keys.intersection(spec.keys())
    _RaiseErrorIfUnexpectedKeys(unexpected_keys, reason='Only allow to specify together with `local-package-path` in the first `--worker-pool-spec` flag')