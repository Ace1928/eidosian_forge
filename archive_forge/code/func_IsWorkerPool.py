from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import re
from apitools.base.protorpclite import messages as proto_messages
from apitools.base.py import encoding as apitools_encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import files
import six
def IsWorkerPool(resource_name):
    """Determine if the provided full resource name is a worker pool.

  Args:
    resource_name: str, The string to test.

  Returns:
    bool, True if the string is a worker pool's full resource name.
  """
    return bool(re.match(WORKERPOOL_NAME_MATCHER, resource_name))