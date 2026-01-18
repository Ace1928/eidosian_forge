from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import files
def GetImageFromPath(path):
    """Builds an Image message from a path.

  Args:
    path: the path arg given to the command.

  Raises:
    ImagePathError: if the image path does not exist and does not seem to be
        a remote URI.

  Returns:
    vision_v1_messages.Image: an image message containing information for the
        API on the image to analyze.
  """
    messages = apis.GetMessagesModule(VISION_API, VISION_API_VERSION)
    image = messages.Image()
    if os.path.isfile(path):
        image.content = files.ReadBinaryFileContents(path)
    elif re.match(IMAGE_URI_FORMAT, path):
        image.source = messages.ImageSource(imageUri=path)
    else:
        raise ImagePathError('The image path does not exist locally or is not properly formatted. A URI for a remote image must be a Google Cloud Storage image URI, which must be in the form `gs://bucket_name/object_name`, or a publicly accessible image HTTP/HTTPS URL. Please double-check your input and try again.')
    return image