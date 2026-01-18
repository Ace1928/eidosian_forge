from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import time
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.container import constants as gke_constants
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import times
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
from cmd argument to set a surge upgrade strategy.
def ResourceLimitsFromFlags(self, options):
    """Create cluster's autoscaling resource limits from command line flags.

    Args:
      options: Either CreateClusterOptions or UpdateClusterOptions.

    Returns:
      Cluster's new autoscaling resource limits.
    """
    new_resource_limits = []
    if options.min_cpu is not None or options.max_cpu is not None:
        new_resource_limits.append(self.messages.ResourceLimit(resourceType='cpu', minimum=options.min_cpu, maximum=options.max_cpu))
    if options.min_memory is not None or options.max_memory is not None:
        new_resource_limits.append(self.messages.ResourceLimit(resourceType='memory', minimum=options.min_memory, maximum=options.max_memory))
    if options.max_accelerator is not None:
        accelerator_type = options.max_accelerator.get('type')
        min_count = 0
        if options.min_accelerator is not None:
            if options.min_accelerator.get('type') != accelerator_type:
                raise util.Error(MISMATCH_ACCELERATOR_TYPE_LIMITS_ERROR_MSG)
            min_count = options.min_accelerator.get('count', 0)
        new_resource_limits.append(self.messages.ResourceLimit(resourceType=options.max_accelerator.get('type'), minimum=min_count, maximum=options.max_accelerator.get('count', 0)))
    return new_resource_limits