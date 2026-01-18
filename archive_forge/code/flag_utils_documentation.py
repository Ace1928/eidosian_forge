from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.recommender import service as recommender_service
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import base
from googlecloudsdk.core import yaml
Get API version string.

  Converts API version string from release track value.

  Args:
    release_track: release_track value, can be ALPHA, BETA, GA

  Returns:
    API version string.
  