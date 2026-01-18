from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.endpoints import config_reporter
from googlecloudsdk.api_lib.endpoints import exceptions
from googlecloudsdk.api_lib.endpoints import services_util
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions as services_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import http_encoding
import six.moves.urllib.parse
def ShowConfigReport(self, service, service_config_id, log_func=log.warning):
    """Run and display results (if any) from the Push Advisor.

    Args:
      service: The name of the service for which to compare configs.
      service_config_id: The new config ID to compare against the active config.
      log_func: The function to which to pass advisory messages
        (default: log.warning).

    Returns:
      The number of advisory messages returned by the Push Advisor.
    """
    num_changes_with_advice = 0
    reporter = config_reporter.ConfigReporter(service)
    reporter.new_config.SetConfigId(service_config_id)
    reporter.old_config.SetConfigUseDefaultId()
    change_report = reporter.RunReport()
    if not change_report or not change_report.configChanges:
        return 0
    changes = change_report.configChanges
    for change in changes:
        if change.advices:
            if num_changes_with_advice < NUM_ADVICE_TO_PRINT:
                log_func('%s\n', services_util.PushAdvisorConfigChangeToString(change))
            num_changes_with_advice += 1
    if num_changes_with_advice > NUM_ADVICE_TO_PRINT:
        log_func('%s total changes with advice found, check config report file for full list.\n', num_changes_with_advice)
    return num_changes_with_advice