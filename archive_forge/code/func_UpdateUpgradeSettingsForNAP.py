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
def UpdateUpgradeSettingsForNAP(self, options, max_surge, max_unavailable):
    """Updates upgrade setting for autoprovisioned node pool."""
    if options.enable_autoprovisioning_surge_upgrade and options.enable_autoprovisioning_blue_green_upgrade:
        raise util.Error('UpgradeSettings must contain only one of: --enable-autoprovisioning-surge-upgrade, --enable-autoprovisioning-blue-green-upgrade')
    upgrade_settings = self.messages.UpgradeSettings()
    upgrade_settings.maxSurge = max_surge
    upgrade_settings.maxUnavailable = max_unavailable
    if options.enable_autoprovisioning_surge_upgrade:
        upgrade_settings.strategy = self.messages.UpgradeSettings.StrategyValueValuesEnum.SURGE
    if options.enable_autoprovisioning_blue_green_upgrade:
        upgrade_settings.strategy = self.messages.UpgradeSettings.StrategyValueValuesEnum.BLUE_GREEN
    if options.autoprovisioning_standard_rollout_policy is not None or options.autoprovisioning_node_pool_soak_duration is not None:
        upgrade_settings.blueGreenSettings = self.UpdateBlueGreenSettingsForNAP(upgrade_settings, options)
    return upgrade_settings