from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
def IsApiVersion(token):
    """Check if the token parsed from Url is API version."""
    versions = ('alpha', 'beta', 'v1', 'v2', 'v3', 'v4', 'dogfood', 'head')
    for api_version in versions:
        if api_version in token:
            return True
    return False