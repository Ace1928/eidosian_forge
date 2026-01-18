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
def _GenerateLogFilePath(dest, instance_id, serial_port_num=None):
    """Gets the full path of the destination file to which to download logs."""
    base_dir = None
    if dest.startswith('~'):
        base_dir = files.ExpandHomeDir(dest)
    elif os.path.isabs(dest):
        base_dir = dest
    else:
        base_dir = files.GetCWD()
    date_str = _GetDateStr()
    file_name = ''
    if serial_port_num:
        file_name = '{}_serial_port_{}_logs_{}.txt'.format(instance_id, serial_port_num, date_str)
    else:
        file_name = '{}_cloud_logging_logs_{}.txt'.format(instance_id, date_str)
    return os.path.join(base_dir, file_name)