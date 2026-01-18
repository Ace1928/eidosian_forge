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
def DownloadInstanceLogs(instance, logs, dest, serial_port_num=None):
    """Downloads the logs and puts them in the specified destination.

  Args:
    instance: the instance from which to download the logs.
    logs: the list of logs from the instance.
    dest: the destination folder.
    serial_port_num: The serial port from which the logs came
  """
    dest_file = _GenerateLogFilePath(dest, instance.id, serial_port_num)
    if serial_port_num:
        contents = logs.contents.split('\n')
        lines_to_download = []
        for line in contents:
            if 'OSConfigAgent' in line:
                lines_to_download.append(line)
        files.WriteFileContents(dest_file, '\n'.join(lines_to_download))
    else:
        formatter = logs_util.LogPrinter()
        formatter.RegisterFormatter(_PayloadFormatter)
        files.WriteFileContents(dest_file, formatter.Format(logs[0]) + '\n')
        with files.FileWriter(dest_file, append=True) as f:
            for log in logs[1:]:
                f.write(formatter.Format(log) + '\n')
    logging.Print('Logs downloaded to {}.'.format(dest_file))