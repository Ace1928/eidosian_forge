from os_ken.services.protocols.bgp.info_base.vrf import VrfRtImportMap
from os_ken.services.protocols.bgp.info_base.vrf4 import Vrf4NlriImportMap
from os_ken.services.protocols.bgp.info_base.vrf6 import Vrf6NlriImportMap
class ImportMapManager(object):

    def __init__(self):
        self._import_maps_by_name = {}

    def create_vpnv4_nlri_import_map(self, name, value):
        self._create_import_map_factory(name, value, Vrf4NlriImportMap)

    def create_vpnv6_nlri_import_map(self, name, value):
        self._create_import_map_factory(name, value, Vrf6NlriImportMap)

    def create_rt_import_map(self, name, value):
        self._create_import_map_factory(name, value, VrfRtImportMap)

    def _create_import_map_factory(self, name, value, cls):
        if self._import_maps_by_name.get(name) is not None:
            raise ImportMapAlreadyExistsError()
        self._import_maps_by_name[name] = cls(value)

    def get_import_map_by_name(self, name):
        return self._import_maps_by_name.get(name)