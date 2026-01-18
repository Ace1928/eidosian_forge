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
def CorrectUpdateMask(ref, args, request):
    """Returns the update request with the corrected mask.

  The API expects a request with an update mask of 'schema', whereas the inline
  schema argument generates an update mask of 'schema.columns'. So if --schema
  was specified, we have to correct the update mask. As for the physical schema,
  the mask must be added.

  Args:
    ref: The entry resource reference.
    args: The parsed args namespace.
    request: The update entry request.

  Returns:
    Request with corrected update mask.
  """
    del ref
    if args.IsSpecified('physical_schema_type'):
        request.updateMask += ',schema'
    if args.IsSpecified('schema'):
        request.updateMask = request.updateMask.replace('schema.columns', 'schema')
    return request