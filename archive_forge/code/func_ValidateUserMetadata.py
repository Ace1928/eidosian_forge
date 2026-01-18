from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import re
import enum
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def ValidateUserMetadata(metadata):
    """Validates if user-specified metadata.

  Checks if it contains values which may conflict with container deployment.
  Args:
    metadata: user-specified VM metadata.

  Raises:
    InvalidMetadataKeyException: if there is conflict with user-provided
    metadata
  """
    for entry in metadata.items:
        if entry.key in [CONTAINER_MANIFEST_KEY, GKE_DOCKER]:
            raise InvalidMetadataKeyException(entry.key)