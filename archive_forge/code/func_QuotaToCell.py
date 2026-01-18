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
def QuotaToCell(region):
    """Formats the metric from the parent function."""
    for quota in region.get('quotas', []):
        if quota.get('metric') != metric:
            continue
        if is_integer:
            return '{0:6}/{1}'.format(int(quota.get('usage')), int(quota.get('limit')))
        else:
            return '{0:7.2f}/{1:.2f}'.format(quota.get('usage'), quota.get('limit'))
    return ''