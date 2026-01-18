from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import base
def GetApiClientForApiVersion(api_version=None):
    if not api_version:
        api_version = core_apis.ResolveVersion(GKEHUB_API_NAME)
    return core_apis.GetClientInstance(GKEHUB_API_NAME, api_version)