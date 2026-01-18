import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class CreateFlavorProfile(neutronV20.CreateCommand):
    """Create a Neutron service flavor profile."""
    resource = 'service_profile'

    def add_known_arguments(self, parser):
        parser.add_argument('--description', help=_('Description for the flavor profile.'))
        parser.add_argument('--driver', help=_('Python module path to driver.'))
        parser.add_argument('--metainfo', help=_('Metainfo for the flavor profile.'))
        utils.add_boolean_argument(parser, '--enabled', default=argparse.SUPPRESS, help=_('Sets enabled flag.'))

    def args2body(self, parsed_args):
        body = {}
        neutronV20.update_dict(parsed_args, body, ['description', 'driver', 'enabled', 'metainfo'])
        return {self.resource: body}