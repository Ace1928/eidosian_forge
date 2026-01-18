from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
Writes a key value pair to the metadata server.

    Args:
      project: The project string the instance is in.
      instance_ref: The instance the metadata server relates to.
      key: The string key to enter the data in.
      val: The string value to be written at the key.
    