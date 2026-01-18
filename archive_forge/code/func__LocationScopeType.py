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
def _LocationScopeType(instance_group):
    """Returns a location scope type, could be region or zone."""
    if 'zone' in instance_group:
        return 'zone'
    elif 'region' in instance_group:
        return 'region'
    else:
        return None