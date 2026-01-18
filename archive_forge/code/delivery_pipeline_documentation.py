from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.clouddeploy import client_util
Creates a rollback for a given target.

    Args:
      name: pipeline name
      request: RollbackTargetRequest

    Returns:
      RollbackTargetResponse
    