from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.active_directory import util
def UpdateOnPremSIDDetails(domain_ref, args, request):
    """Generate Migrating Domain Details."""
    onprem_arg = args.onprem_domains
    disable_sid_domains = args.disable_sid_filtering_domains or []
    messages = util.GetMessagesForResource(domain_ref)
    on_prem_dets = []
    for name in onprem_arg or []:
        disable_sid_filter = False
        if name in disable_sid_domains:
            disable_sid_filter = True
        onprem_req = messages.OnPremDomainDetails(domainName=name, disableSidFiltering=disable_sid_filter)
        on_prem_dets.append(onprem_req)
    request.enableMigrationRequest = messages.EnableMigrationRequest(migratingDomains=on_prem_dets)
    return request