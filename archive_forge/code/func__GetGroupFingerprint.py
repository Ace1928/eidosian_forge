from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _GetGroupFingerprint(compute_client, group_ref):
    """Gets fingerprint of given instance group."""
    compute = compute_client.apitools_client
    if IsZonalGroup(group_ref):
        service = compute.instanceGroups
        request = compute.MESSAGES_MODULE.ComputeInstanceGroupsGetRequest(project=group_ref.project, zone=group_ref.zone, instanceGroup=group_ref.instanceGroup)
    else:
        service = compute.regionInstanceGroups
        request = compute.MESSAGES_MODULE.ComputeRegionInstanceGroupsGetRequest(project=group_ref.project, region=group_ref.region, instanceGroup=group_ref.instanceGroup)
    errors = []
    resources = compute_client.MakeRequests(requests=[(service, 'Get', request)], errors_to_collect=errors)
    if errors:
        utils.RaiseException(errors, FingerprintFetchException, error_message='Could not set named ports for resource:')
    return resources[0].fingerprint