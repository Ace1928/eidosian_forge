from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
Ensure that test-service endpoints are compatible with each other.

  A staging/test ToolResults API endpoint will not work correctly with a
  production Testing API endpoint, and vice versa. This check is only relevant
  for internal development.

  Raises:
    IncompatibleApiEndpointsError if the endpoints are not compatible.
  