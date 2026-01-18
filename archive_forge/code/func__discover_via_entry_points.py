from stevedore import extension
from neutronclient.neutron import v2_0 as neutronV20
def _discover_via_entry_points():
    emgr = extension.ExtensionManager('neutronclient.extension', invoke_on_load=False)
    return ((ext.name, ext.plugin) for ext in emgr)