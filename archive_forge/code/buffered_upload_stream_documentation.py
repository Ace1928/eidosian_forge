from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.command_lib.storage import upload_stream
Indicates that this stream is not seekable.

    Needed so that boto3 can correctly identify how to treat this stream.
    The library attempts to seek to the beginning after an upload completes,
    which is not always possible.

    Apitools does not check the return value of this method, so it will not
    raise issues for resumable uploads.

    Returns:
      False
    