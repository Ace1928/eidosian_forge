from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import property_selector
import six
import six.moves.http_client
def _BackendsToCell(backend_service):
    """Comma-joins the names of the backend services."""
    return ','.join((backend.get('group') for backend in backend_service.get('backends', [])))