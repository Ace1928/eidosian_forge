from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.resource_manager import org_policies
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def _GetPolicyNoThrow(project_id, errors_to_propagate):
    """Call GetPolicy and handle possible errors from backend."""
    try:
        return _GetPolicy(project_id)
    except apitools_exceptions.HttpError as e:
        errors_to_propagate.append(e)
        return None