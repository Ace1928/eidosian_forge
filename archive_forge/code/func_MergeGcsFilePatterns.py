from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from os import path
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.data_catalog import entries_v1
from googlecloudsdk.api_lib.data_catalog import util as api_util
from googlecloudsdk.command_lib.concepts import exceptions as concept_exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def MergeGcsFilePatterns(ref, args, request):
    """Merges user-provided GCS file patterns with existing patterns.

  Args:
    ref: The entry resource reference.
    args: The parsed args namespace.
    request: The update entry request.

  Returns:
    Request with merged GCS file pattern.
  """
    if not _IsChangeFilePatternSpecified(args):
        return request
    del ref
    entry_ref = args.CONCEPTS.entry.Parse()
    file_patterns = entries_v1.EntriesClient().Get(entry_ref).gcsFilesetSpec.filePatterns or []
    if args.IsSpecified('clear_file_patterns'):
        file_patterns = []
    if args.IsSpecified('remove_file_patterns'):
        to_remove = set(args.remove_file_patterns)
        file_patterns = [b for b in file_patterns if b not in to_remove]
    if args.IsSpecified('add_file_patterns'):
        file_patterns += args.add_file_patterns
    arg_utils.SetFieldInMessage(request, 'googleCloudDatacatalogV1Entry.gcsFilesetSpec.filePatterns', file_patterns)
    request.updateMask += ',gcsFilesetSpec.filePatterns'
    return request