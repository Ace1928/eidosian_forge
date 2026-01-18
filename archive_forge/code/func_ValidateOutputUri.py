from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from six.moves import urllib
def ValidateOutputUri(output_uri):
    """Validates given output URI against validator function.

  Args:
    output_uri: str, the output URI for the analysis.

  Raises:
    UriFormatError: if the URI is not valid.

  Returns:
    str, The same output_uri.
  """
    if output_uri and (not storage_util.ObjectReference.IsStorageUrl(output_uri)):
        raise UriFormatError(OUTPUT_ERROR_MESSAGE.format(output_uri))
    return output_uri