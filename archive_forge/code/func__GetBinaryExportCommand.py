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
def _GetBinaryExportCommand(self, args, command_name, resource_uri=None, skip_parent=False, skip_filter=False):
    """Constructs and returns a list representing the binary export command."""
    cmd = [self._export_service, '--oauth2-token', self._GetToken(), command_name]
    if command_name == 'export':
        if not resource_uri:
            raise ClientException('`_GetBinaryExportCommand` requires a resource uri for export commands.')
        cmd.extend([resource_uri])
    if command_name == 'bulk-export':
        cmd.extend(['--on-error', getattr(args, 'on_error', 'ignore')])
        if not skip_parent:
            if args.IsSpecified('organization'):
                cmd.extend(['--organization', args.organization])
            elif args.IsSpecified('folder'):
                cmd.extend(['--folder', args.folder])
            else:
                project = args.project or properties.VALUES.core.project.GetOrFail()
                cmd.extend(['--project', project])
        if not skip_filter:
            if args.IsSpecified('resource_types') or args.IsSpecified('resource_types_file'):
                cmd.extend(['--resource-types', self._ParseResourceTypes(args)])
    if getattr(args, 'storage_path', None):
        cmd.extend(['--storage-key', args.storage_path])
    if getattr(args, 'resource_format', None):
        cmd.extend(['--resource-format', _NormalizeResourceFormat(args.resource_format)])
        if args.resource_format == 'terraform':
            cmd.extend(['--iam-format', 'none'])
    if self._OutputToFileOrDir(args.path):
        cmd.extend(['--output', args.path])
    return cmd