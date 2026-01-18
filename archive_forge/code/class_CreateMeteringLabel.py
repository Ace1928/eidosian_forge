from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
class CreateMeteringLabel(neutronv20.CreateCommand):
    """Create a metering label for a given tenant."""
    resource = 'metering_label'

    def add_known_arguments(self, parser):
        parser.add_argument('name', metavar='NAME', help=_('Name of the metering label to be created.'))
        parser.add_argument('--description', help=_('Description of the metering label to be created.'))
        parser.add_argument('--shared', action='store_true', help=_('Set the label as shared.'))

    def args2body(self, parsed_args):
        body = {'name': parsed_args.name}
        neutronv20.update_dict(parsed_args, body, ['tenant_id', 'description', 'shared'])
        return {'metering_label': body}