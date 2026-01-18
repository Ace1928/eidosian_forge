from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronv20
class AddTag(neutronv20.NeutronCommand):
    """Add a tag into the resource."""

    def get_parser(self, prog_name):
        parser = super(AddTag, self).get_parser(prog_name)
        _add_common_arguments(parser)
        parser.add_argument('--tag', required=True, help=_('Tag to be added.'))
        return parser

    def take_action(self, parsed_args):
        client = self.get_client()
        if not parsed_args.tag:
            raise exceptions.CommandError(_('Cannot add an empty value as tag'))
        resource_type, resource_id = _convert_resource_args(client, parsed_args)
        client.add_tag(resource_type, resource_id, parsed_args.tag)