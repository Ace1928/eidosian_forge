from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os
import re
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.app import logs_util
from googlecloudsdk.api_lib.logging import common
from googlecloudsdk.api_lib.logging import util as logging_util
from googlecloudsdk.core import log as logging
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def CheckCloudLogs(project, instance):
    """Checks the Cloud logs created by this instance for errors."""
    logging.Print('The troubleshooter is now fetching and analyzing logs...\n')
    cloud_logging_enabled = False
    for account in instance.serviceAccounts:
        if 'https://www.googleapis.com/auth/logging.write' in account.scopes:
            cloud_logging_enabled = True
            break
    if not cloud_logging_enabled:
        logging.Print('Cloud logging is not enabled for this project.')
        return False
    filter_str = 'resource.type="gce_instance" AND resource.labels.instance_id="{}" AND log_name="projects/{}/logs/OSConfigAgent"'.format(instance.id, project.name)
    logs = list(common.FetchLogs(filter_str, limit=1000, order_by='DESC'))
    logs.reverse()
    severity_enum = logging_util.GetMessages().LogEntry.SeverityValueValuesEnum
    error_log_counter = 0
    earliest_timestamp = None
    for log in logs:
        if log.severity >= severity_enum.ERROR:
            error_log_counter += 1
        if not earliest_timestamp:
            earliest_timestamp = log.timestamp
    if logs:
        response_message = 'The troubleshooter analyzed Cloud Logging logs and found:\n'
        response_message += '> {} OSConfigAgent log entries.\n'.format(len(logs))
        response_message += '> Among them, {} {} errors.\n'.format(error_log_counter, 'has' if error_log_counter == 1 else 'have')
        response_message += '> The earliest timestamp is ' + (earliest_timestamp if earliest_timestamp else 'N/A') + '.'
        logging.Print(response_message)
        cont = console_io.PromptContinue(prompt_string='Download all OSConfigAgent logs?')
        if cont:
            dest = console_io.PromptWithDefault(message='Destination folder for log download', default='~/Downloads/osconfig-logs/')
            logging.Print('Downloading log entries...')
            DownloadInstanceLogs(instance, logs, six.text_type(dest))
    else:
        logging.Print('The troubleshooter analyzed Cloud Logging logs and found no logs.')
        return False
    return True