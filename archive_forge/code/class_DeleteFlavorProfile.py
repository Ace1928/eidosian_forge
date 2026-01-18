import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class DeleteFlavorProfile(neutronV20.DeleteCommand):
    """Delete a given Neutron service flavor profile."""
    resource = 'service_profile'