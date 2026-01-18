from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def CheckAddressPoolNameUniqueness(external_lb_address_pools):
    """Checks for unique address pool names in the given list of dictionaries.

  Args:
    external_lb_address_pools: A list of dictionaries representing
    ExternalLoadBalancerPool messages.

  Returns:
    str: An error message if a duplicate address pool name is found,
    otherwise None.
  """
    address_pool_set = set()
    for pool in external_lb_address_pools:
        if 'addressPool' in pool and pool['addressPool']:
            if pool['addressPool'] in address_pool_set:
                return f'Duplicate address pool name: {pool['addressPool']}'
            address_pool_set.add(pool['addressPool'])
    return None