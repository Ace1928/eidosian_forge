from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib import network_security
from googlecloudsdk.api_lib import network_services
from googlecloudsdk.core import resources
def UrlInReferences(url, references):
    return bool(list(filter(lambda ref: CompareUrlRelativeReferences(url, ref), references)))