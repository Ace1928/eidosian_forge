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
def UpdateBlueGreenSettingsForNAP(self, upgrade_settings, options):
    """Update blue green settings field in upgrade_settings for autoprovisioned node pool.
    """
    blue_green_settings = upgrade_settings.blueGreenSettings or self.messages.BlueGreenSettings()
    if options.autoprovisioning_node_pool_soak_duration is not None:
        blue_green_settings.nodePoolSoakDuration = options.autoprovisioning_node_pool_soak_duration
    if options.autoprovisioning_standard_rollout_policy is not None:
        standard_rollout_policy = blue_green_settings.standardRolloutPolicy or self.messages.StandardRolloutPolicy()
        if 'batch-node-count' in options.autoprovisioning_standard_rollout_policy and 'batch-percent' in options.autoprovisioning_standard_rollout_policy:
            raise util.Error('Autoprovisioning StandardRolloutPolicy must contain only one of: batch-node-count, batch-percent')
        standard_rollout_policy.batchPercentage = standard_rollout_policy.batchNodeCount = None
        if 'batch-node-count' in options.autoprovisioning_standard_rollout_policy:
            standard_rollout_policy.batchNodeCount = options.autoprovisioning_standard_rollout_policy['batch-node-count']
        elif 'batch-percent' in options.autoprovisioning_standard_rollout_policy:
            standard_rollout_policy.batchPercentage = options.autoprovisioning_standard_rollout_policy['batch-percent']
        if 'batch-soak-duration' in options.autoprovisioning_standard_rollout_policy:
            standard_rollout_policy.batchSoakDuration = options.autoprovisioning_standard_rollout_policy['batch-soak-duration']
        blue_green_settings.standardRolloutPolicy = standard_rollout_policy
    return blue_green_settings