from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def _parse_common_args(body, parsed_args):
    neutronV20.update_dict(parsed_args, body, ['name', 'description'])