from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import files
def GetGcsSourceFromPath(input_file):
    """Validate a Google Cloud Storage location to read the PDF/TIFF file from.

  Args:
    input_file: the input file path arg given to the command.

  Raises:
    GcsSourceError: if the file is not a Google Cloud Storage object.

  Returns:
    vision_v1_messages.GcsSource: Google Cloud Storage URI for the input file.
    This must only be a Google Cloud Storage object.
    Wildcards are not currently supported.
  """
    messages = apis.GetMessagesModule(VISION_API, VISION_API_VERSION)
    gcs_source = messages.GcsSource()
    if re.match(FILE_URI_FORMAT, input_file):
        gcs_source.uri = input_file
    else:
        raise GcsSourceError('The URI for the input file must be a Google Cloud Storage object, which must be in the form `gs://bucket_name/object_name.Please double-check your input and try again.')
    return gcs_source