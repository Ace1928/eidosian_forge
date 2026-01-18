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
def ValidateClusterAutoscaling(self, autoscaling, for_update):
    """Validate cluster autoscaling configuration.

    Args:
      autoscaling: autoscaling configuration to be validated.
      for_update: Is function executed for update operation.

    Raises:
      Error if the new configuration is invalid.
    """
    if autoscaling.enableNodeAutoprovisioning:
        if not for_update or autoscaling.resourceLimits:
            cpu_found = any((limit.resourceType == 'cpu' for limit in autoscaling.resourceLimits))
            mem_found = any((limit.resourceType == 'memory' for limit in autoscaling.resourceLimits))
            if not cpu_found or not mem_found:
                raise util.Error(NO_AUTOPROVISIONING_LIMITS_ERROR_MSG)
            defaults = autoscaling.autoprovisioningNodePoolDefaults
            if defaults:
                if defaults.upgradeSettings:
                    max_surge_found = defaults.upgradeSettings.maxSurge is not None
                    max_unavailable_found = defaults.upgradeSettings.maxUnavailable is not None
                    if max_unavailable_found != max_surge_found:
                        raise util.Error(BOTH_AUTOPROVISIONING_UPGRADE_SETTINGS_ERROR_MSG)
                if defaults.management:
                    auto_upgrade_found = defaults.management.autoUpgrade is not None
                    auto_repair_found = defaults.management.autoRepair is not None
                    if auto_repair_found != auto_upgrade_found:
                        raise util.Error(BOTH_AUTOPROVISIONING_MANAGEMENT_SETTINGS_ERROR_MSG)
                if defaults.shieldedInstanceConfig:
                    secure_boot_found = defaults.shieldedInstanceConfig.enableSecureBoot is not None
                    integrity_monitoring_found = defaults.shieldedInstanceConfig.enableIntegrityMonitoring is not None
                    if secure_boot_found != integrity_monitoring_found:
                        raise util.Error(BOTH_AUTOPROVISIONING_SHIELDED_INSTANCE_SETTINGS_ERROR_MSG)
    elif autoscaling.resourceLimits:
        raise util.Error(LIMITS_WITHOUT_AUTOPROVISIONING_MSG)
    elif autoscaling.autoprovisioningNodePoolDefaults and (autoscaling.autoprovisioningNodePoolDefaults.serviceAccount or autoscaling.autoprovisioningNodePoolDefaults.oauthScopes or autoscaling.autoprovisioningNodePoolDefaults.management or autoscaling.autoprovisioningNodePoolDefaults.upgradeSettings):
        raise util.Error(DEFAULTS_WITHOUT_AUTOPROVISIONING_MSG)