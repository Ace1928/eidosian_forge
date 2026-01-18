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
def _ValidateWorkerPoolSpecsFromConfig(job_spec):
    """Validate WorkerPoolSpec message instances imported from the config file."""
    for spec in job_spec.workerPoolSpecs:
        use_python_package = spec.pythonPackageSpec and (spec.pythonPackageSpec.executorImageUri or spec.pythonPackageSpec.pythonModule)
        use_container = spec.containerSpec and spec.containerSpec.imageUri
        if use_container and use_python_package or (not use_container and (not use_python_package)):
            raise exceptions.InvalidArgumentException('--config', 'Exactly one of fields [pythonPackageSpec, containerSpec] is required for a [workerPoolSpecs] in the YAML config file.')