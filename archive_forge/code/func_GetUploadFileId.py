from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import zipfile
from googlecloudsdk.api_lib import apigee
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.util import archive
from googlecloudsdk.core.util import files
from six.moves import urllib
def GetUploadFileId(upload_url):
    """Helper function to extract the upload file id from the signed URL.

  Archive deployments must be uploaded to a provided signed URL in the form of:
  https://storage.googleapis.com/<bucket id>/<file id>.zip?<additional headers>
  This function extracts the file id from the URL (e.g., <file id>.zip).

  Args:
    upload_url: A string of the signed URL.

  Returns:
    A string of the file id.
  """
    url = urllib.parse.urlparse(upload_url)
    split_path = url.path.split('/')
    return split_path[-1]