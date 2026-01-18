from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.transfer import creds_util
from googlecloudsdk.command_lib.transfer import jobs_flag_util
from googlecloudsdk.command_lib.transfer import name_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
def _create_or_modify_object_conditions(transfer_spec, args, messages):
    """Creates or modifies ObjectConditions based on args."""
    if not (getattr(args, 'include_prefixes', None) or getattr(args, 'exclude_prefixes', None) or getattr(args, 'include_modified_before_absolute', None) or getattr(args, 'include_modified_after_absolute', None) or getattr(args, 'include_modified_before_relative', None) or getattr(args, 'include_modified_after_relative', None)):
        return
    if not transfer_spec.objectConditions:
        transfer_spec.objectConditions = messages.ObjectConditions()
    if getattr(args, 'include_prefixes', None):
        transfer_spec.objectConditions.includePrefixes = args.include_prefixes
    if getattr(args, 'exclude_prefixes', None):
        transfer_spec.objectConditions.excludePrefixes = args.exclude_prefixes
    if getattr(args, 'include_modified_before_absolute', None):
        modified_before_datetime_string = args.include_modified_before_absolute.astimezone(times.UTC).isoformat()
        transfer_spec.objectConditions.lastModifiedBefore = modified_before_datetime_string
    if getattr(args, 'include_modified_after_absolute', None):
        modified_after_datetime_string = args.include_modified_after_absolute.astimezone(times.UTC).isoformat()
        transfer_spec.objectConditions.lastModifiedSince = modified_after_datetime_string
    if getattr(args, 'include_modified_before_relative', None):
        transfer_spec.objectConditions.minTimeElapsedSinceLastModification = '{}s'.format(args.include_modified_before_relative)
    if getattr(args, 'include_modified_after_relative', None):
        transfer_spec.objectConditions.maxTimeElapsedSinceLastModification = '{}s'.format(args.include_modified_after_relative)