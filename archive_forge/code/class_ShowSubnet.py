import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ShowSubnet(neutronV20.ShowCommand):
    """Show information of a given subnet."""
    resource = 'subnet'