import collections
import itertools
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib.i18n import _
from osc_lib import utils
from oslo_utils import excutils
from osc_placement.resources import common
from osc_placement import version
class SetClassInventory(command.ShowOne):
    """Replace the inventory record of the class for the resource provider.

    Example::

        openstack resource provider inventory class set <uuid> VCPU             --total 16             --max_unit 4             --reserved 1
    """

    def get_parser(self, prog_name):
        parser = super(SetClassInventory, self).get_parser(prog_name)
        parser.add_argument('uuid', metavar='<uuid>', help='UUID of the resource provider')
        parser.add_argument('resource_class', metavar='<class>', help=RC_HELP)
        for name, props in INVENTORY_FIELDS.items():
            parser.add_argument('--' + name, metavar='<{}>'.format(name), required=props['required'], type=props['type'], help=props['help'])
        return parser

    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = RP_BASE_URL + '/' + parsed_args.uuid
        rp = http.request('GET', url).json()
        payload = {'resource_provider_generation': rp['generation']}
        for field in FIELDS:
            value = getattr(parsed_args, field, None)
            if value is not None:
                payload[field] = value
        url = PER_CLASS_URL.format(uuid=parsed_args.uuid, resource_class=parsed_args.resource_class)
        resource = http.request('PUT', url, json=payload).json()
        return (FIELDS, utils.get_dict_properties(resource, FIELDS))