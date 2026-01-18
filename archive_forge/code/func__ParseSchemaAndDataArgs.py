from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import os
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions as c_except
def _ParseSchemaAndDataArgs(self, source, prefix=None, skip_foreign_keys=False, source_profile=None, target=None, target_profile=None, write_limit=None, dry_run=False, log_level=None, **kwargs):
    """"Parse args for the schema-and-data command."""
    del kwargs
    exec_args = ['schema-and-data']
    if source:
        exec_args.extend(['--source', source])
    if prefix:
        exec_args.extend(['--prefix', prefix])
    if skip_foreign_keys:
        exec_args.append('--skip-foreign-keys')
    if source_profile:
        exec_args.extend(['--source-profile', source_profile])
    if target:
        exec_args.extend(['--target', target])
    if target_profile:
        exec_args.extend(['--target-profile', target_profile])
    if write_limit:
        exec_args.extend(['--write-limit', write_limit])
    if dry_run:
        exec_args.append('--dry-run')
    if log_level:
        exec_args.extend(['--log-level', log_level])
    return exec_args