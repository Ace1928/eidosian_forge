from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import copy
import json
import os
from googlecloudsdk.command_lib.anthos import flags
from googlecloudsdk.command_lib.anthos.common import file_parsers
from googlecloudsdk.command_lib.anthos.common import messages
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions as c_except
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import requests
import six
from six.moves import urllib
class AnthosCliWrapper(binary_operations.StreamingBinaryBackedOperation):
    """Binary operation wrapper for anthoscli commands."""

    def __init__(self, **kwargs):
        custom_errors = {'MISSING_EXEC': messages.MISSING_BINARY.format(binary='anthoscli')}
        super(AnthosCliWrapper, self).__init__(binary='anthoscli', custom_errors=custom_errors, **kwargs)

    def _ParseGetArgs(self, repo_uri, local_dest, file_pattern=None, **kwargs):
        del kwargs
        exec_args = ['get', repo_uri, local_dest]
        if file_pattern:
            exec_args.extend(['--pattern', file_pattern])
        return exec_args

    def _ParseUpdateArgs(self, local_dir, repo_uri=None, strategy=None, dry_run=False, verbose=False, **kwargs):
        del kwargs
        exec_args = ['update', local_dir]
        if repo_uri:
            exec_args.extend(['--repo', repo_uri])
        if dry_run:
            exec_args.append('--dry-run')
        if strategy:
            exec_args.extend(['--strategy', strategy])
        if verbose:
            exec_args.append('--verbose')
        return exec_args

    def _ParseDescribeArgs(self, local_dir, **kwargs):
        del kwargs
        return ['desc', local_dir]

    def _ParseTags(self, tags):
        return ','.join(['{}={}'.format(x, y) for x, y in six.iteritems(tags)])

    def _ParseInitArgs(self, local_dir, description=None, name=None, tags=None, info_url=None, **kwargs):
        del kwargs
        package_path = local_dir
        if not package_path.endswith('/'):
            package_path += '/'
        exec_args = ['init', package_path]
        if description:
            exec_args.extend(['--description', description])
        if name:
            exec_args.extend(['--name', name])
        if tags:
            exec_args.extend(['--tag', self._ParseTags(tags)])
        if info_url:
            exec_args.extend(['--url', info_url])
        return exec_args

    def _ParseApplyArgs(self, apply_dir, project, **kwargs):
        del kwargs
        exec_args = ['apply', '-f', apply_dir, '--project', project]
        return exec_args

    def _ParseExportArgs(self, cluster, project, location, output_dir, **kwargs):
        del kwargs
        exec_args = ['export', '-c', cluster, '--project', project]
        if location:
            exec_args.extend(['--location', location])
        if output_dir:
            exec_args.extend(['--output-directory', output_dir])
        return exec_args

    def _ParseArgsForCommand(self, command, **kwargs):
        if command == 'get':
            return self._ParseGetArgs(**kwargs)
        if command == 'update':
            return self._ParseUpdateArgs(**kwargs)
        if command == 'desc':
            return self._ParseDescribeArgs(**kwargs)
        if command == 'init':
            return self._ParseInitArgs(**kwargs)
        if command == 'apply':
            return self._ParseApplyArgs(**kwargs)
        if command == 'export':
            return self._ParseExportArgs(**kwargs)
        raise binary_operations.InvalidOperationForBinary('Invalid Operation [{}] for anthoscli'.format(command))