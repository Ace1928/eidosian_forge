from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_printer
def GetZoneFromInstance(instance, resource_registry):
    zone_ref = resource_registry.Parse(instance.zone, collection='compute.zones')
    return zone_ref.Name()