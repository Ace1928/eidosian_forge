from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import resources
def LiensClient():
    return apis.GetClientInstance('cloudresourcemanager', LIENS_API_VERSION)