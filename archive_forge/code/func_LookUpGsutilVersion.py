from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
import re
import textwrap
import sys
import gslib
from gslib.utils.system_util import IS_OSX
from gslib.exception import CommandException
from gslib.storage_url import StorageUrlFromString
from gslib.utils.constants import GSUTIL_PUB_TARBALL
from gslib.utils.constants import GSUTIL_PUB_TARBALL_PY2
def LookUpGsutilVersion(gsutil_api, url_str):
    """Looks up the gsutil version of the specified gsutil tarball URL.

  Version is specified in the metadata field set on that object.

  Args:
    gsutil_api: gsutil Cloud API to use when retrieving gsutil tarball.
    url_str: tarball URL to retrieve (such as 'gs://pub/gsutil.tar.gz').

  Returns:
    Version string if URL is a cloud URL containing x-goog-meta-gsutil-version
    metadata, else None.
  """
    url = StorageUrlFromString(url_str)
    if url.IsCloudUrl():
        obj = gsutil_api.GetObjectMetadata(url.bucket_name, url.object_name, provider=url.scheme, fields=['metadata'])
        if obj.metadata and obj.metadata.additionalProperties:
            for prop in obj.metadata.additionalProperties:
                if prop.key == 'gsutil_version':
                    return prop.value