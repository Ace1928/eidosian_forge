from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
Get the list of supported Cloud Build locations.

  Args:
    project: The project to search.

  Returns:
    A CloudbuildProjectsLocationsListRequest object.
  