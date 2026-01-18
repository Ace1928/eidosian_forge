from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.cloudbuild import snapshot
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_transform
def GetDefaultStagingBucket():
    """Returns the default bucket stage files.

  Returns:
    GCS bucket name.
  """
    safe_project = properties.VALUES.core.project.Get(required=True).replace(':', '_').replace('.', '_').replace('google', 'elgoog')
    return safe_project + '_cloudbuild'