from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import zone_utils
from googlecloudsdk.api_lib.compute.regions import utils as region_utils
def WarnAboutScopeDeprecations(ips_refs, client):
    """Tests to check if the zone is deprecated."""
    zone_resource_fetcher = zone_utils.ZoneResourceFetcher(client)
    zone_resource_fetcher.WarnForZonalCreation((ref for ref in ips_refs if ref.Collection() == 'compute.zoneInstantSnapshots'))
    region_resource_fetcher = region_utils.RegionResourceFetcher(client)
    region_resource_fetcher.WarnForRegionalCreation((ref for ref in ips_refs if ref.Collection() == 'compute.regionInstantSnapshots'))