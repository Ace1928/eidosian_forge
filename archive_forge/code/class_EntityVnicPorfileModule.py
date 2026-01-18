from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class EntityVnicPorfileModule(BaseModule):

    def __init__(self, *args, **kwargs):
        super(EntityVnicPorfileModule, self).__init__(*args, **kwargs)

    def _get_dcs_service(self):
        return self._connection.system_service().data_centers_service()

    def _get_dcs_id(self):
        return get_id_by_name(self._get_dcs_service(), self.param('data_center'))

    def _get_network_id(self):
        networks_service = self._get_dcs_service().service(self._get_dcs_id()).networks_service()
        return get_id_by_name(networks_service, self.param('network'))

    def _get_qos_id(self):
        if self.param('qos'):
            qoss_service = self._get_dcs_service().service(self._get_dcs_id()).qoss_service()
            return get_id_by_name(qoss_service, self.param('qos')) if self.param('qos') else None
        return None

    def _get_network_filter_id(self):
        nf_service = self._connection.system_service().network_filters_service()
        return get_id_by_name(nf_service, self.param('network_filter')) if self.param('network_filter') else None

    def _get_network_filter(self):
        network_filter = None
        if self.param('network_filter') == '' or self.param('pass_through') == 'enabled':
            network_filter = otypes.NetworkFilter()
        elif self.param('network_filter'):
            network_filter = otypes.NetworkFilter(id=self._get_network_filter_id())
        return network_filter

    def _get_qos(self):
        qos = None
        if self.param('qos') == '' or self.param('pass_through') == 'enabled':
            qos = otypes.Qos()
        elif self.param('qos'):
            qos = otypes.Qos(id=self._get_qos_id())
        return qos

    def _get_port_mirroring(self):
        if self.param('pass_through') == 'enabled':
            return False
        return self.param('port_mirroring')

    def _get_migratable(self):
        if self.param('migratable') is not None:
            return self.param('migratable')
        if self.param('pass_through') == 'enabled':
            return True

    def build_entity(self):
        return otypes.VnicProfile(name=self.param('name'), network=otypes.Network(id=self._get_network_id()), description=self.param('description') if self.param('description') is not None else None, pass_through=otypes.VnicPassThrough(mode=otypes.VnicPassThroughMode(self.param('pass_through'))) if self.param('pass_through') else None, custom_properties=[otypes.CustomProperty(name=cp.get('name'), regexp=cp.get('regexp'), value=str(cp.get('value'))) for cp in self.param('custom_properties') if cp] if self.param('custom_properties') else None, migratable=self._get_migratable(), qos=self._get_qos(), port_mirroring=self._get_port_mirroring(), network_filter=self._get_network_filter())

    def update_check(self, entity):

        def check_custom_properties():
            if self.param('custom_properties'):
                current = []
                if entity.custom_properties:
                    current = [(cp.name, cp.regexp, str(cp.value)) for cp in entity.custom_properties]
                passed = [(cp.get('name'), cp.get('regexp'), str(cp.get('value'))) for cp in self.param('custom_properties') if cp]
                return sorted(current) == sorted(passed)
            return True
        pass_through = getattr(entity.pass_through.mode, 'name', None)
        return check_custom_properties() and self._get_network_filter_id() == getattr(entity.network_filter, 'id', None) and (self._get_qos_id() == getattr(entity.qos, 'id', None)) and equal(self.param('migratable'), getattr(entity, 'migratable', None)) and equal(self.param('pass_through'), pass_through.lower() if pass_through else None) and equal(self.param('description'), entity.description) and equal(self.param('port_mirroring'), getattr(entity, 'port_mirroring', None))