from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import time
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import logs as cloudbuild_logs
from googlecloudsdk.api_lib.util import requests
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from six.moves import range  # pylint: disable=redefined-builtin
def GetBuildProp(build_op, prop_key, required=False):
    """Extract the value of a build's prop_key from a build operation.

  Args:
    build_op: A Google Cloud Builder build operation.
    prop_key: str, The property name.
    required: If True, raise an OperationError if prop_key isn't present.

  Returns:
    The corresponding build operation value indexed by prop_key.

  Raises:
    OperationError: The required prop_key was not found.
  """
    if build_op.metadata is not None:
        for prop in build_op.metadata.additionalProperties:
            if prop.key == 'build':
                for build_prop in prop.value.object_value.properties:
                    if build_prop.key == prop_key:
                        string_value = build_prop.value.string_value
                        return string_value or build_prop.value
    if required:
        raise OperationError('Build operation does not contain required property [{}]'.format(prop_key))