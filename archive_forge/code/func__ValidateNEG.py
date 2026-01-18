from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import network_endpoint_groups
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.network_endpoint_groups import flags
from googlecloudsdk.core import log
def _ValidateNEG(self, args, neg_ref):
    """Validate NEG input before making request."""
    is_zonal = hasattr(neg_ref, 'zone')
    is_regional = hasattr(neg_ref, 'region')
    network_endpoint_type = args.network_endpoint_type
    valid_scopes = collections.OrderedDict()
    if self.support_port_mapping_neg:
        valid_scopes['gce-vm-ip-port'] = ['zonal', 'regional']
    else:
        valid_scopes['gce-vm-ip-port'] = ['zonal']
    valid_scopes['internet-ip-port'] = ['global', 'regional']
    valid_scopes['internet-fqdn-port'] = ['global', 'regional']
    valid_scopes['serverless'] = ['regional']
    valid_scopes['private-service-connect'] = ['regional']
    valid_scopes['non-gcp-private-ip-port'] = ['zonal']
    valid_scopes['gce-vm-ip'] = ['zonal']
    valid_scopes_inverted = _Invert(valid_scopes)
    if not is_regional and args.client_port_mapping_mode:
        raise exceptions.InvalidArgumentException('--client-port-mapping-mode', 'Client port mapping mode is only supported for regional NEGs.')
    if is_zonal:
        valid_zonal_types = valid_scopes_inverted['zonal']
        if network_endpoint_type not in valid_zonal_types:
            raise exceptions.InvalidArgumentException('--network-endpoint-type', 'Zonal NEGs only support network endpoints of type {0}.{1}'.format(_JoinWithOr(valid_zonal_types), _GetValidScopesErrorMessage(network_endpoint_type, valid_scopes)))
    elif is_regional:
        valid_regional_types = valid_scopes_inverted['regional']
        if network_endpoint_type not in valid_regional_types:
            raise exceptions.InvalidArgumentException('--network-endpoint-type', 'Regional NEGs only support network endpoints of type {0}.{1}'.format(_JoinWithOr(valid_regional_types), _GetValidScopesErrorMessage(network_endpoint_type, valid_scopes)))
        if network_endpoint_type == 'private-service-connect' and (not args.psc_target_service):
            raise exceptions.InvalidArgumentException('--private-service-connect', 'Network endpoint type private-service-connect must specify --psc-target-service for private service NEG.')
    else:
        valid_global_types = valid_scopes_inverted['global']
        if network_endpoint_type not in valid_global_types:
            raise exceptions.InvalidArgumentException('--network-endpoint-type', 'Global NEGs only support network endpoints of type {0}.{1}'.format(_JoinWithOr(valid_global_types), _GetValidScopesErrorMessage(network_endpoint_type, valid_scopes)))
    if is_regional and network_endpoint_type == 'gce-vm-ip-port' and (not args.client_port_mapping_mode):
        raise exceptions.InvalidArgumentException('--client-port-mapping-mode', 'Network endpoint type gce-vm-ip-port must specify --client-port-mapping-mode for regional NEG.')