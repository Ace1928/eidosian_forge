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
def _StatusToCell(zone_or_region):
    """Returns status of a machine with deprecation information if applicable."""
    deprecated = zone_or_region.get('deprecated', '')
    if deprecated:
        return '{0} ({1})'.format(zone_or_region.get('status'), deprecated.get('state'))
    else:
        return zone_or_region.get('status')