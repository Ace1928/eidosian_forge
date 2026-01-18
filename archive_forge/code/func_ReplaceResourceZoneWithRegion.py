from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from apitools.base.py import encoding
from googlecloudsdk.core import exceptions
def ReplaceResourceZoneWithRegion(ref, args, request):
    """Replaces the request.name 'locations/{zone}' with 'locations/{region}'."""
    del ref, args
    request.name = re.sub('(projects/[-a-z0-9]+/locations/[a-z]+-[a-z]+[0-9])[-a-z0-9]*((?:/.*)?)', '\\1\\2', request.name)
    return request