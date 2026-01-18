from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import List, Optional
from apitools.base.py import encoding
from googlecloudsdk.command_lib.run.integrations.formatters import base
from googlecloudsdk.command_lib.run.integrations.formatters import states
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages as runapps
def _GetSSLStatuses(self, resource_components: List[runapps.ResourceComponentStatus], resource: runapps.Resource):
    ssl_cert_components = self._FindAllComponentsByType(resource_components, 'google_compute_managed_ssl_certificate')
    statuses = []
    for component in ssl_cert_components:
        gussed_domain = self._GuessDomainFromSSLComponentName(component.name)
        matched_domain = None
        for domain_config in resource.subresources:
            res_domain = encoding.MessageToDict(domain_config.config).get('domain', '')
            if gussed_domain == res_domain:
                matched_domain = res_domain
            elif res_domain.startswith(gussed_domain) and matched_domain is None:
                matched_domain = res_domain
        if matched_domain is None:
            matched_domain = gussed_domain
        comp_state = str(component.state) if component.state else states.UNKNOWN
        statuses.append((matched_domain, comp_state))
    return statuses