import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import availability_zone
def _format_external_gateway_info(router):
    try:
        return jsonutils.dumps(router['external_gateway_info'])
    except (TypeError, KeyError):
        return ''