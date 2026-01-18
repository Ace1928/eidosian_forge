from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import errno
import io
import os
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.command_lib.asset import utils as asset_utils
from googlecloudsdk.command_lib.util.resource_map.declarative import resource_name_translator
from googlecloudsdk.core import exceptions as c_except
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.util import files
import six
def _CallBulkExport(self, cmd, args, asset_list_input=None):
    """Execute actual bulk-export command on config-connector binary."""
    if self._OutputToFileOrDir(args.path):
        self._TryCreateOutputPath(args.path)
        preexisting_file_count = sum([len(files_in_dir) for r, d, files_in_dir in os.walk(args.path)])
        with progress_tracker.ProgressTracker(message='Exporting resource configurations to [{}]'.format(args.path), aborted_message='Aborted Export.'):
            exit_code, _, error_value = _ExecuteBinary(cmd=cmd, in_str=asset_list_input)
        if exit_code != 0:
            if 'Error 403' in error_value:
                msg = 'Permission denied during export. Please ensure the Cloud Asset Inventory API is enabled.'
                if args.storage_path:
                    msg += ' Also check that Cloud IAM permissions are set for `--storage-path` [{}]'.format(args.storage_path)
                raise ClientException(msg)
            else:
                raise ClientException('Error executing export:: [{}]'.format(error_value))
        else:
            _BulkExportPostStatus(preexisting_file_count, args.path)
        return exit_code
    else:
        log.status.write('Exporting resource configurations to stdout...\n')
        return _ExecuteBinaryWithStreaming(cmd=cmd, in_str=asset_list_input)