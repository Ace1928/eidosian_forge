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
def _LocationName(instance_group):
    """Returns a location name, could be region name or zone name."""
    if 'zone' in instance_group:
        return path_simplifier.Name(instance_group['zone'])
    elif 'region' in instance_group:
        return path_simplifier.Name(instance_group['region'])
    else:
        return None