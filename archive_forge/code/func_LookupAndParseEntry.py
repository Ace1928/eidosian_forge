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
def LookupAndParseEntry(ref, args, request):
    """Parses the entry into the request, performing a lookup first if necessary.

  Args:
    ref: None.
    args: The parsed args namespace.
    request: The update entry request.

  Returns:
    Request containing the parsed entry.
  Raises:
    UnderSpecifiedEntryError: if ENTRY was only partially specified.
    RequiredMutexGroupError: if both or neither ENTRY, --lookup-entry specified.
  """
    del ref
    entry_ref = args.CONCEPTS.entry.Parse()
    if args.IsSpecified('entry') and (not entry_ref):
        raise UnderSpecifiedEntryError('Argument [ENTRY : --entry-group=ENTRY_GROUP --location=LOCATION] was not fully specified.')
    if entry_ref and args.IsSpecified('lookup_entry') or (not entry_ref and (not args.IsSpecified('lookup_entry'))):
        raise concept_exceptions.RequiredMutexGroupError('entry', '([ENTRY : --entry-group=ENTRY_GROUP --location=LOCATION] | --lookup-entry)')
    if entry_ref:
        request.name = entry_ref.RelativeName()
    else:
        client = entries_v1.EntriesClient()
        request.name = client.Lookup(args.lookup_entry).name
    return request