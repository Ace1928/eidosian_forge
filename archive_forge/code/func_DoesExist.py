from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.api_gateway import base
from googlecloudsdk.command_lib.api_gateway import common_flags
def DoesExist(self, api_ref):
    """Checks if an Api object exists.

    Args:
      api_ref: Resource, a resource reference for the api

    Returns:
      Boolean, indicating whether or not exists
    """
    try:
        self.Get(api_ref)
    except apitools_exceptions.HttpNotFoundError:
        return False
    return True