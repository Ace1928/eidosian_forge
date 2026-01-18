from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronv20
def _add_common_arguments(parser):
    parser.add_argument('--resource-type', choices=TAG_RESOURCES, dest='resource_type', required=True, help=_('Resource Type.'))
    parser.add_argument('--resource', required=True, help=_('Resource name or ID.'))