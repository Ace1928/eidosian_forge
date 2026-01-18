import abc
from neutron_lib.api.definitions import portbindings
@property
def _supports_port_binding(self):
    return self.__class__.bind_port != MechanismDriver.bind_port