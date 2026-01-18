from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import files
def GetGcsDestinationFromPath(path):
    """Validate a Google Cloud Storage location to write the output to.

  Args:
    path: the path arg given to the command.

  Raises:
    GcsDestinationError: if the file is not a Google Cloud Storage object.

  Returns:
    vision_v1_messages.GcsDestination:Google Cloud Storage URI prefix
    where the results will be stored.
    This must only be a Google Cloud Storage object.
    Wildcards are not currently supported.
  """
    messages = apis.GetMessagesModule(VISION_API, VISION_API_VERSION)
    gcs_destination = messages.GcsDestination()
    if re.match(FILE_PREFIX, path):
        gcs_destination.uri = path
    else:
        raise GcsDestinationError('The URI for the input file must be a Google Cloud Storage object, which must be in the File prefix format `gs://bucket_name/here/file_name_prefix.or directory prefix format `gs://bucket_name/some/location/. Please double-check your input and try again.')
    return gcs_destination