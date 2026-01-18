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
def CheckSerialLogOutput(client, project, instance, zone):
    """Checks the serial log output of the given instance for errors."""
    logging.Print('The troubleshooter is now checking serial log output.')
    logs_to_download = []
    serial_logs = []
    for port in range(1, _NUM_SERIAL_PORTS + 1):
        serial_log = None
        num_errors = 0
        try:
            serial_log = _GetSerialLogOutput(client, project, instance, zone, port)
            num_errors = len(re.findall('OSConfigAgent Error', serial_log.contents))
        except exceptions.Error:
            num_errors = None
        serial_logs.append(serial_log)
        if num_errors is not None:
            logging.Print('Port {}: {} OSConfigAgent error(s)'.format(port, num_errors))
            if num_errors:
                logs_to_download.append(port)
        else:
            logging.Print('Port {}: N/A'.format(port))
    if logs_to_download:
        cont = console_io.PromptContinue(prompt_string='Download all OSConfigAgent logs?')
        if cont:
            dest = console_io.PromptWithDefault(message='Destination folder for log download (default is ~/Downloads/osconfig-logs):', default='~/Downloads/osconfig-logs')
            logging.Print('Downloading serial log entries...')
            for port in logs_to_download:
                DownloadInstanceLogs(instance, serial_logs[port - 1], six.text_type(dest), serial_port_num=port)