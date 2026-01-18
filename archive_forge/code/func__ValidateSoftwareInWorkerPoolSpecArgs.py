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
def _ValidateSoftwareInWorkerPoolSpecArgs(worker_pool_specs):
    """Validates the software fields specified in all `--worker-pool-spec` flags."""
    has_local_package = _ValidateSoftwareInFirstWorkerPoolSpec(worker_pool_specs[0])
    if len(worker_pool_specs) > 1:
        _ValidateSoftwareInRestWorkerPoolSpecs(worker_pool_specs[1:], has_local_package)