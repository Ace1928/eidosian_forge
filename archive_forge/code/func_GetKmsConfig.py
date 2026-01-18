from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def GetKmsConfig(self, kmsconfig_ref):
    """Get Cloud NetApp KMS Config information."""
    request = self.messages.NetappProjectsLocationsKmsConfigsGetRequest(name=kmsconfig_ref.RelativeName())
    return self.client.projects_locations_kmsConfigs.Get(request)