import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class UpdateSubnet(neutronV20.UpdateCommand):
    """Update subnet's information."""
    resource = 'subnet'

    def add_known_arguments(self, parser):
        add_updatable_arguments(parser)

    def args2body(self, parsed_args):
        body = {}
        updatable_args2body(parsed_args, body, for_create=False)
        return {'subnet': body}