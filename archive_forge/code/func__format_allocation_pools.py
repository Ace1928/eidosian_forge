import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def _format_allocation_pools(subnet):
    try:
        return '\n'.join([jsonutils.dumps(pool) for pool in subnet['allocation_pools']])
    except (TypeError, KeyError):
        return ''