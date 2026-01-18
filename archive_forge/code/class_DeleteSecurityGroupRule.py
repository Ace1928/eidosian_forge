import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class DeleteSecurityGroupRule(neutronV20.DeleteCommand):
    """Delete a given security group rule."""
    resource = 'security_group_rule'
    allow_names = False