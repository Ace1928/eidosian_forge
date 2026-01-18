from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.accesscontextmanager import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import log
from googlecloudsdk.core import resources as core_resources
import six
def _CreateServicePerimeterConfig(messages, mask_prefix, resources, restricted_services, levels, vpc_allowed_services, enable_vpc_accessible_services, ingress_policies=None, egress_policies=None):
    """Returns a ServicePerimeterConfig and its update mask."""
    config = messages.ServicePerimeterConfig()
    mask = []
    _SetIfNotNone('resources', resources, config, mask)
    _SetIfNotNone('restrictedServices', restricted_services, config, mask)
    _SetIfNotNone('ingressPolicies', ingress_policies, config, mask)
    _SetIfNotNone('egressPolicies', egress_policies, config, mask)
    if levels is not None:
        mask.append('accessLevels')
        level_names = []
        for l in levels:
            if isinstance(l, six.string_types):
                level_names.append(l)
            else:
                level_names.append(l.RelativeName())
            config.accessLevels = level_names
    if enable_vpc_accessible_services is not None or vpc_allowed_services is not None:
        service_filter = messages.VpcAccessibleServices()
        service_filter_mask = []
        _SetIfNotNone('allowedServices', vpc_allowed_services, service_filter, service_filter_mask)
        _SetIfNotNone('enableRestriction', enable_vpc_accessible_services, service_filter, service_filter_mask)
        config.vpcAccessibleServices = service_filter
        mask.extend(['vpcAccessibleServices.' + m for m in service_filter_mask])
    if not mask:
        return (None, [])
    return (config, ['{}.{}'.format(mask_prefix, item) for item in mask])