from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
class ShowAgent(neutronV20.ShowCommand):
    """Show information of a given agent."""
    resource = 'agent'
    allow_names = False
    json_indent = 5