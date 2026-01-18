import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def _format_host_routes(subnet):
    try:
        return '\n'.join([jsonutils.dumps(route) for route in subnet['host_routes']])
    except (TypeError, KeyError):
        return ''