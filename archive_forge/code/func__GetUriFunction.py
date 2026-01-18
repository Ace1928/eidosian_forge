from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.dns import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dns import flags
from googlecloudsdk.core import properties
def _GetUriFunction(api_version):

    def _GetUri(resource):
        return util.GetRegistry(api_version).Create('dns.managedZones', project=properties.VALUES.core.project.GetOrFail, managedZone=resource.name).SelfLink()
    return _GetUri