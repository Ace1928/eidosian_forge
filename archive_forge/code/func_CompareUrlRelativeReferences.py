from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib import network_security
from googlecloudsdk.api_lib import network_services
from googlecloudsdk.core import resources
def CompareUrlRelativeReferences(url1, url2):
    """Compares relative resource references (skips namespace)."""
    return url1.split('projects')[1] == url2.split('projects')[1]