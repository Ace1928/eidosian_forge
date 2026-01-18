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
def ValidateCreateArgs(args, job_spec_from_config, version):
    """Validate the argument values specified in `create` command."""
    if args.worker_pool_spec:
        _ValidateWorkerPoolSpecArgs(args.worker_pool_spec, version)
    else:
        _ValidateWorkerPoolSpecsFromConfig(job_spec_from_config)