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
def RemoveMaintenanceExclusion(self, cluster_ref, existing_policy, exclusion_name):
    """Removes a maintenance exclusion from the maintenance policy by name."""
    existing_policy = self._NormalizeMaintenanceExclusionsForPolicy(existing_policy)
    existing_exclusions = self._GetMaintenanceExclusionNames(existing_policy)
    if exclusion_name not in existing_exclusions:
        message = 'No maintenance exclusion with name {0} exists. Existing exclusions: {1}.'.format(exclusion_name, ', '.join(existing_exclusions))
        raise util.Error(message)
    props = []
    for ex in existing_policy.window.maintenanceExclusions.additionalProperties:
        if ex.key != exclusion_name:
            props.append(ex)
    existing_policy.window.maintenanceExclusions.additionalProperties = props
    return self._SendMaintenancePolicyRequest(cluster_ref, existing_policy)